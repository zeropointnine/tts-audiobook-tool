/**
 * Reads and decodes the custom tts-audiobook-tool metadata from a FLAC or MP4 file
 * Returns parsed hash object from the metadata, or an error string
 */
async function loadAppMetadata(file, url) {

    FLAC_FIELD = "TTS_AUDIOBOOK_TOOL"
    MP4_MEAN = "tts-audiobook-tool"
    MP4_TAG = "audiobook-data"

    const isFlac = url ? url.toLowerCase().endsWith("flac") : file.name.toLowerCase().endsWith("flac");

    if (url) {
        // Create File-like adapter
        file = new RemoteFileLike(url)
    }

    // Read "abr" metadata (json string)
    let tagValue = null;
    try {
        if (isFlac) {
            tagValue = await findCustomFlacTag(file, FLAC_FIELD);
        } else {
            // Will assume is mp4/m4a
            tagValue = await findCustomMp4Tag(file, MP4_MEAN, MP4_TAG);
        }
    } catch (error) {
        return "Error parsing data:" + error;
    } finally {
        // "Closing the file stream" is handled by the browser implicitly
        // when the File/Blob objects are no longer referenced and garbage collected.
        // There's no explicit close() method for File objects in the browser.
    }
    if (!tagValue) {
        return "File has no ABR metadata";
    }

    // Parse json, validate
    try {
        o = JSON.parse(tagValue)
    } catch (e) {
        return "Couldn't parse metadata: " + e
    }
    if (!o) {
        return "Couldn't parse metadata"
    }

    // Text segments
    const timedTextSegments = o["text_segments"];
    if (!timedTextSegments || timedTextSegments.length == 0) {
        return "ABR metadata missing required field 'text_segments'";
    }

    // Raw text (no longer required but)
    let rawText = ""
    const rawTextBase64 = o["raw_text"]
    if (!rawTextBase64) {
        console.warning("empty or missing raw_text field")
    } else {
        const binaryStr = atob(rawTextBase64.replace(/_/g, '/').replace(/-/g, '+'));
        const bytes = new Uint8Array(binaryStr.length);
        for (let i = 0; i < binaryStr.length; i++) {
            bytes[i] = binaryStr.charCodeAt(i);
        }
        // Decompress
        const decompressed = pako.inflate(bytes); // zlib decompression
        rawText = new TextDecoder('utf-8').decode(decompressed);
        if (!rawText) {
            console.warning("decoded rawText is empty?");
        }
    }
    if (!rawText) {
        rawText = "";
    }

    result = {
        "raw_text": rawText,
        "text_segments": timedTextSegments
    }
    return result
}

/**
 * Returns value or null, or throws error.
 */
async function findCustomMp4Tag(file, targetMean, targetTagName) {

    const textDecoder = new TextDecoder('utf-8');
    const CHUNK_SIZE = 1024 * 64; // 64KB chunks for reading

    async function readChunk(offset, length) {
        if (offset < 0) throw new Error("readChunk: start offset cannot be negative.");
        const end = offset + length;
        let effectiveLength = length;

        if (offset >= file.size) {
            console.debug(`readChunk: Attempt to read at or beyond EOF. Offset: ${offset}, File size: ${file.size}`);
            return null; // Or throw error, depending on desired strictness
        }

        if (end > file.size) {
            effectiveLength = file.size - offset;
            console.debug(`readChunk: Adjusted read length to EOF. Original: ${length}, Effective: ${effectiveLength}`);
        }
        if (effectiveLength <= 0) {
            console.debug(`readChunk: Effective length is zero or negative. Offset: ${offset}, File size: ${file.size}`);
            return null;
        }

        const slice = file.slice(offset, offset + effectiveLength);
        const arrayBuffer = await slice.arrayBuffer();
        return new DataView(arrayBuffer);
    }

    async function parseAtoms(currentOffset, maxOffset, path = []) {
        let offset = currentOffset;
        while (offset < maxOffset && offset < file.size) {
            if (offset + 8 > file.size) { // Need at least 8 bytes for size and type
                console.debug(`parseAtoms: Not enough data for atom header at offset ${offset}. Path: ${path.join('/')}`);
                break;
            }
            const headerView = await readChunk(offset, 8);
            if (!headerView || headerView.byteLength < 8) {
                console.debug(`parseAtoms: Failed to read atom header or insufficient data at offset ${offset}. Path: ${path.join('/')}`);
                break;
            }

            let atomSize = headerView.getUint32(0, false); // Big-endian
            const atomTypeBuffer = headerView.buffer.slice(4, 8);
            const atomType = textDecoder.decode(atomTypeBuffer);
            // console.debug(`Atom: Type=${atomType}, Size=${atomSize}, Offset=${offset}, Path: ${path.join('/')}`);

            let atomDataOffset = offset + 8;
            let atomDataSize = atomSize - 8;

            if (atomSize === 1) { // 64-bit size
                if (offset + 16 > file.size) {
                    // console.warn(`parseAtoms: Not enough data for 64-bit atom size at offset ${offset}. Path: ${path.join('/')}`);
                    break;
                }
                const sizeView = await readChunk(offset + 8, 8);
                if (!sizeView || sizeView.byteLength < 8) {
                     // console.warn(`parseAtoms: Failed to read 64-bit atom size at offset ${offset}. Path: ${path.join('/')}`);
                    break;
                }
                // JavaScript doesn't handle 64-bit integers natively well,
                // but for file parsing, we assume sizes fit in Number.MAX_SAFE_INTEGER
                atomSize = Number(sizeView.getBigUint64(0, false));
                atomDataOffset = offset + 16;
                atomDataSize = atomSize - 16;
            } else if (atomSize === 0) { // Atom extends to end of file (or enclosing atom)
                atomSize = maxOffset - offset;
                atomDataSize = atomSize - 8;
                 // console.debug(`Atom ${atomType} extends to end of current scope. New Size=${atomSize}`);
            }

            if (atomSize < 8 && atomSize !==0 && atomSize !==1) { // Minimum size for type and size fields
                throw new Error(`Invalid atom size ${atomSize} for type ${atomType} at offset ${offset}. Path: ${path.join('/')}`);
            }
            if (atomDataOffset + atomDataSize > file.size) {
                console.warn(`Atom ${atomType} at offset ${offset} with size ${atomSize} extends beyond file size ${file.size}. Truncating. Path: ${path.join('/')}`);
                atomDataSize = file.size - atomDataOffset;
                if(atomDataSize < 0) atomDataSize = 0;
            }

            const newPath = [...path, atomType];

            if (atomType === 'moov' || atomType === 'udta' || atomType === 'meta' || atomType === 'ilst' || atomType.startsWith('©') || atomType === '----') {
                if (atomType === 'meta') {
                    // The 'meta' atom has a 4-byte version/flags field after the type
                    atomDataOffset += 4;
                    atomDataSize -= 4;
                }
                if (atomType === '----') {
                    // This is the custom tag container
                    if (atomDataSize > 0) {
                        const dashContentView = await readChunk(atomDataOffset, atomDataSize);
                        if (dashContentView) {
                            const foundValue = await parseDashAtomContent(dashContentView, targetMean, targetTagName, newPath);
                            if (foundValue !== null) return foundValue;
                        }
                    }
                } else {
                    // Recursively parse container atoms
                    const result = await parseAtoms(atomDataOffset, atomDataOffset + atomDataSize, newPath);
                    if (result !== null) return result;
                }
            }
            // else, skip unknown/uninteresting atom content

            offset += atomSize;
            if (atomSize === 0) { // Should not happen if 'atom extends to end' was handled correctly
                // console.warn("Atom size was 0, breaking loop to prevent infinite loop. This might indicate a parsing issue or malformed file.");
                break;
            }
        }
        return null; // Target not found in this branch
    }

    async function parseDashAtomContent(dashContentDataView, expectedMean, expectedTagName, path) {
        // A '----' atom contains sub-atoms: 'mean', 'name', and 'data'
        let actualMean = null;
        let actualName = null;
        let dataPayload = null;
        let subOffset = 0;

        // console.debug(`Parsing '----' atom content. Size: ${dashContentDataView.byteLength}. Path: ${path.join('/')}`);

        while (subOffset < dashContentDataView.byteLength) {
            if (subOffset + 8 > dashContentDataView.byteLength) {
                console.debug(`----: Not enough data for sub-atom header at offset ${subOffset}. Path: ${path.join('/')}`);
                break;
            }
            const subAtomSize = dashContentDataView.getUint32(subOffset, false);
            const subAtomTypeBuffer = dashContentDataView.buffer.slice(dashContentDataView.byteOffset + subOffset + 4, dashContentDataView.byteOffset + subOffset + 8);
            const subAtomType = textDecoder.decode(subAtomTypeBuffer);

            // console.debug(`---- Sub-atom: Type=${subAtomType}, Size=${subAtomSize}, Offset=${subOffset}. Path: ${path.join('/')}`);

            if (subAtomSize < 8) {
                throw new Error(`Invalid sub-atom size ${subAtomSize} for type ${subAtomType} at offset ${subOffset}. Path: ${path.join('/')}`);
            }

            const subAtomDataOffset = subOffset + 8;
            let subAtomDataLength = subAtomSize - 8;

            if (subAtomDataOffset + subAtomDataLength > dashContentDataView.byteLength) {
                console.warn(`----: Sub-atom ${subAtomType} extends beyond parent '----' atom. Truncating. Path: ${path.join('/')}`);
                subAtomDataLength = dashContentDataView.byteLength - subAtomDataOffset;
                if(subAtomDataLength < 0) subAtomDataLength = 0;
            }

            // The 'mean' and 'name' atoms have a 4-byte version/flags field
            const contentStart = subAtomDataOffset + 4; // Skip version (1 byte) and flags (3 bytes)
            const contentLength = subAtomDataLength - 4;

            if (contentLength < 0) {
                 console.warn(`----: Sub-atom ${subAtomType} content length is negative after accounting for version/flags. Path: ${path.join('/')}`);
                 subOffset += subAtomSize;
                 continue;
            }

            if (subAtomType === 'mean') {
                const meanSlice = new Uint8Array(dashContentDataView.buffer, dashContentDataView.byteOffset + contentStart, contentLength);
                actualMean = textDecoder.decode(meanSlice);
                // console.debug(`---- Found mean: '${actualMean}'. Path: ${path.join('/')}`);
            } else if (subAtomType === 'name') {
                const nameSlice = new Uint8Array(dashContentDataView.buffer, dashContentDataView.byteOffset + contentStart, contentLength);
                actualName = textDecoder.decode(nameSlice);
                // console.debug(`---- Found name: '${actualName}'. Path: ${path.join('/')}`);
            } else if (subAtomType === 'data') {
                // Data atom: type (4 bytes), version (1), flags (3), locale (4), data
                // The first 8 bytes of data atom content are type indicator and locale.
                // Actual string data starts after that.
                if (subAtomDataLength >= 8) { // 4 bytes for type/version, 4 for locale
                    const dataIndicator = dashContentDataView.getUint32(subAtomDataOffset, false); // type + version/flags
                    // const localeIndicator = dashContentDataView.getUint32(subAtomDataOffset + 4, false);

                    // Type '1' indicates UTF-8 encoded string.
                    // Others could be integers, images, etc. We only care about UTF-8 strings.
                    if ((dataIndicator >> 24) === 1 || dataIndicator === 1) { // Check if the type code (first byte of dataIndicator) is 1 for UTF-8
                        const dataSlice = new Uint8Array(dashContentDataView.buffer, dashContentDataView.byteOffset + subAtomDataOffset + 8, subAtomDataLength - 8);
                        dataPayload = textDecoder.decode(dataSlice);
                        // console.debug(`---- Found data (UTF-8): '${dataPayload.substring(0,100)}...'. Path: ${path.join('/')}`);
                    } else {
                        console.debug(`---- Data atom is not UTF-8 (type ${dataIndicator >> 24}), skipping. Path: ${path.join('/')}`);
                    }
                } else {
                    console.debug(`---- Data atom too short for content. Length: ${subAtomDataLength}. Path: ${path.join('/')}`);
                }
            }
            subOffset += subAtomSize;
        }

        if (actualMean === expectedMean && actualName === expectedTagName && dataPayload !== null) {
            // console.log(`Found target MP4 tag: mean='${actualMean}', name='${actualName}'. Path: ${path.join('/')}`);
            return dataPayload;
        }
        return null;
    }

    // Start parsing from the beginning of the file
    return await parseAtoms(0, file.size);
}

/**
 * Using the given file handle,
 * returns a FLAC file's metadata value for the given tag name,
 * or null if error or not found.
 *
 */
async function findCustomFlacTag(file, tagName) {

    let offset = 0;
    const textDecoder = new TextDecoder('utf-8');

    // Helper function to read a chunk of the file
    async function readChunk(start, length) {
        if (start + length > file.size) {
            throw new Error("Attempting to read beyond file end.");
        }
        const slice = file.slice(start, start + length);
        const arrayBuffer = await slice.arrayBuffer();
        return new DataView(arrayBuffer);
    }

    // Confirm "fLaC" marker (4 bytes)

    let dataView = await readChunk(offset, 4);
    const marker = textDecoder.decode(dataView.buffer);
    if (marker !== "fLaC") {
        throw new Error("Missing 'fLaC' marker");
    }
    offset += 4;

    // Loop through metadata blocks

    let isLastMetadataBlock = false;

    while (!isLastMetadataBlock) {

        if (offset + 4 > file.size) {
            throw new Error("Unexpected end of file while reading metadata block header");
        }

        // Read metadata block header (4 bytes)
        dataView = await readChunk(offset, 4);
        offset += 4;

        const headerByte1 = dataView.getUint8(0);
        isLastMetadataBlock = (headerByte1 & 0x80) !== 0; // MSB
        const blockType = headerByte1 & 0x7F; // 7 LSBs

        // Three bytes, big-endian
        const blockLength = (dataView.getUint8(1) << 16) | (dataView.getUint8(2) << 8) | dataView.getUint8(3);

        // console.log(`Metadata Block: Type=${blockType}, Length=${blockLength}, IsLast=${isLastMetadataBlock}`);

        if (blockType === 4) { // VORBIS_COMMENT block

            // console.log("Found VORBIS_COMMENT block. Parsing...");

            if (offset + blockLength > file.size) {
                throw new Error("VORBIS_COMMENT block length exceeds file size.");
            }

            const vorbisCommentDataView = await readChunk(offset, blockLength);
            const foundValue = parseVorbisCommentBlock(vorbisCommentDataView, tagName, textDecoder);

            if (foundValue !== null) {
                // Tag found, return its value
                return foundValue;
            }

            // If not found in this block, continue
            // (though typically there's only one VORBIS_COMMENT block)
        }

        // Move to the next block
        offset += blockLength;

        if (offset > file.size && !isLastMetadataBlock) {
             throw new Error("File ended prematurely before last metadata block flag.");
        }

         if (offset === file.size && !isLastMetadataBlock) {
            // This can happen if the last block exactly fills the file
            // but isn't marked as last, or if audio data is missing.
            // For our purpose of finding tags, if we haven't found it by now, it's not there.
            console.warn("Reached end of file, but last metadata block flag was not set. Assuming no more metadata.");
            break;
        }
    }

    // Target tag not found after checking all relevant blocks
    return null;
}

function parseVorbisCommentBlock(dataView, targetTagName, textDecoder) {
    let currentPosition = 0; // Position within this VORBIS_COMMENT block's data

    // --- Vendor String ---
    // Vendor length (4 bytes, Little Endian)
    if (currentPosition + 4 > dataView.byteLength) throw new Error("Vorbis block too short for vendor length.");
    const vendorLength = dataView.getUint32(currentPosition, true); // true for little-endian
    currentPosition += 4;

    // Skip vendor string
    if (currentPosition + vendorLength > dataView.byteLength) throw new Error("Vorbis block too short for vendor string.");
    // const vendorString = textDecoder.decode(new Uint8Array(dataView.buffer, dataView.byteOffset + currentPosition, vendorLength));
    // console.log("Vendor String:", vendorString);
    currentPosition += vendorLength;

    // --- User Comment List ---
    // Number of comments (4 bytes, Little Endian)
    if (currentPosition + 4 > dataView.byteLength) throw new Error("Vorbis block too short for comment count.");
    const numComments = dataView.getUint32(currentPosition, true);
    currentPosition += 4;
    // console.log(`Number of comments: ${numComments}`);

    for (let i = 0; i < numComments; i++) {
        // Length of comment (4 bytes, Little Endian)
        if (currentPosition + 4 > dataView.byteLength) throw new Error(`Vorbis block too short for comment ${i} length.`);
        const commentLength = dataView.getUint32(currentPosition, true);
        // console.log("comment length", commentLength)
        currentPosition += 4;

        // Comment string (UTF-8)
        if (currentPosition + commentLength > dataView.byteLength) throw new Error(`Vorbis block too short for comment ${i} data.`);
        const commentBytes = new Uint8Array(dataView.buffer, dataView.byteOffset + currentPosition, commentLength);
        const commentString = textDecoder.decode(commentBytes);
        currentPosition += commentLength;

        // console.log("Comment:", commentString);
        const [partOne, ...rest] = commentString.split("=");
        if (partOne.toUpperCase() === targetTagName.toUpperCase() && rest.length > 0) {
            const partTwo = rest.join("=")
            // console.log(`Found target tag "${targetTagName}" with value "${partTwo}"`);
            return partTwo; // Return the value of the tag
        }

    }
    return null; // Target tag not found in this block
}

function escapeHtml(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#39;")
}

function splitWhitespace(str) {
    // Match leading whitespace, content, and trailing whitespace
    str = str || "";
    const match = str.match(/^(\s*)(.*?)(\s*)$/);
    return {
      before: match[1],
      content: match[2],
      after: match[3]
    };
}

function msToString(milliseconds) {
    const totalSeconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    if (minutes > 0) {
        return `${minutes}m${seconds}s`;
    } else {
        return `${seconds}s`;
    }
}

function isTouchDevice() {
    // hand-wavey
    return !matchMedia('(pointer:fine)').matches
}

function hasPersistentKeyboard() {
    // hand-wavey test
    const hasFinePointer = window.matchMedia('(pointer: fine)').matches;
    const hasHoverSupport = window.matchMedia('(hover: hover)').matches;
    return (hasFinePointer && hasHoverSupport);
}

// -------------------------------------------------------------------------------------------------

/**
 * Adapter to use urls like Files.
 *
 * For compatibility reasons, it has a "size" property, but is spurious (returns max int).
 * This is done to prevent triggering elevated CORS-related requirements
 * (which happens when you make a header request for file size).
 * It is the responsibility of the client to take account of this limitation, basically.
 */
class RemoteFileLike {

    constructor(url) {
        this.url  = url;
        this.size = Number.MAX_SAFE_INTEGER;
    }

    slice(start = 0, end = this.size) {
        return new RemoteBlob(this, start, end);
    }

    async readRange(start, end) {
        // end === Infinity or > real length → turn into open-range request
        const rangeVal = end === Infinity || end === this.size
            ? `bytes=${start}-`
            : `bytes=${start}-${end - 1}`;

        const res = await fetch(this.url, {headers: {Range: rangeVal}});
        if (!res.ok) {
            // 416 can happen when start is past EOF; treat as empty
            if (res.status === 416) return new ArrayBuffer(0);
            throw new Error(`Range request failed: ${res.status}`);
        }
        return res.arrayBuffer();
    }
}

/**
 * RemoteFileLike helper class
 */
class RemoteBlob {
    constructor(remote, start, end) {
        this.remote = remote;
        this.start  = Math.max(0, start);
        this.end    = end;
    }

    async arrayBuffer() {
        return this.remote.readRange(this.start, this.end);
    }

    async text() {
        const ab = await this.arrayBuffer();
        return new TextDecoder().decode(ab);
    }

    async* stream(chunk = 64 * 1024) {
        let offset = this.start;
        while (true) {
            const next = offset + chunk;
            const buf  = await this.remote.readRange(offset, next);
            if (buf.byteLength === 0) break;   // EOF
            yield new Uint8Array(buf);
            offset += buf.byteLength;
        }
    }
}
