/**
 * Reads and decodes the custom tts-audiobook-tool metadata from a FLAC file
 * and returns an object
 */

async function loadMetadataFromAppFlac(fileOrUrl) {

    let file; // Blob/File object

    if (typeof fileOrUrl === 'string' && (fileOrUrl.startsWith('http://') || fileOrUrl.startsWith('https://'))) {
        try {
            const response = await fetch(fileOrUrl);
            if (!response.ok) {
                console.error(`HTTP error, status ${response.status}`);
                return null;
            }
            file = await response.blob();
        } catch (error) {
            console.error("Error fetching URL:", error);
            return null;
        }
    } else if (fileOrUrl instanceof File || fileOrUrl instanceof Blob) {
        file = fileOrUrl;
    } else {
        console.error("Invalid input: expected File, Blob, or URL string.");
        return null;
    }

    if (!file) {
        return null;
    }

    let tagValue = null
    try {
        tagValue = await findCustomFlacTag(file, "TTS_AUDIOBOOK_TOOL");
    } catch (error) {
        console.error("Error parsing FLAC:", error);
        return null
    } finally {
        // "Closing the file stream" is handled by the browser implicitly
        // when the File/Blob objects are no longer referenced and garbage collected.
        // There's no explicit close() method for File objects in the browser.
    }

    try {
        o = JSON.parse(tagValue)
    } catch (e) {
        console.error(e)
        return null
    }

    const timedTextSegments = o["text_segments"]
    if (!timedTextSegments) {
        console.error("missing text_segments")
        return null
    }
    if (timedTextSegments.length == 0) {
        console.error("text_segments is empty")
        return null
    }

    const rawTextBase64 = o["raw_text"]
    if (!rawTextBase64) {
        console.error("empty or missing raw_text field")
        return null
    }
    const binaryStr = atob(rawTextBase64.replace(/_/g, '/').replace(/-/g, '+'));
    const bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) {
        bytes[i] = binaryStr.charCodeAt(i);
    }
    // Decompress
    const decompressed = pako.inflate(bytes); // zlib decompression
    const rawText = new TextDecoder('utf-8').decode(decompressed);
    if (!rawText) {
        console.error("decoded rawText is empty?")
        return null
    }

    result = {
        "raw_text": rawText,
        "text_segments": timedTextSegments
    }
    return result
}

async function findCustomFlacTag(file, targetTagName) {
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

    // 1. Check for "fLaC" marker (4 bytes)
    let dataView = await readChunk(offset, 4);
    const marker = textDecoder.decode(dataView.buffer);
    if (marker !== "fLaC") {
        throw new Error("Not a valid FLAC file (missing 'fLaC' marker).");
    }
    offset += 4;
    // console.log("Found 'fLaC' marker.");

    // 2. Loop through metadata blocks
    let isLastMetadataBlock = false;
    while (!isLastMetadataBlock) {
        if (offset + 4 > file.size) {
            throw new Error("Unexpected end of file while reading metadata block header.");
        }
        // Read metadata block header (4 bytes)
        dataView = await readChunk(offset, 4);
        offset += 4;

        const headerByte1 = dataView.getUint8(0);
        isLastMetadataBlock = (headerByte1 & 0x80) !== 0; // MSB
        const blockType = headerByte1 & 0x7F;       // 7 LSBs

        // Length is 3 bytes, Big Endian
        const blockLength = (dataView.getUint8(1) << 16) | (dataView.getUint8(2) << 8) | dataView.getUint8(3);

        // console.log(`Metadata Block: Type=${blockType}, Length=${blockLength}, IsLast=${isLastMetadataBlock}`);

        if (blockType === 4) { // VORBIS_COMMENT block
            // console.log("Found VORBIS_COMMENT block. Parsing...");
            if (offset + blockLength > file.size) {
                throw new Error("VORBIS_COMMENT block length exceeds file size.");
            }
            const vorbisCommentDataView = await readChunk(offset, blockLength);
            const foundValue = parseVorbisCommentBlock(vorbisCommentDataView, targetTagName, textDecoder);
            if (foundValue !== null) {
                return foundValue; // Tag found, return its value
            }
            // If not found in this block, continue (though typically there's only one VORBIS_COMMENT block)
        }

        offset += blockLength; // Move to the next block

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

    return null; // Target tag not found after checking all relevant blocks
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
         .replace(/&/g, "&")
         .replace(/</g, "<")
         .replace(/>/g, ">")
         .replace(/"/g, "\"")
         .replace(/'/g, "'")
}
