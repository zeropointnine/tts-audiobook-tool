"use strict";

/**
 * Manages and displays the "book text.""
 * 
 * Uses multiple callbacks to interact with <audio> element, etc.
 */
class BookText {

    // TODO: 
    //  Too many callbacks. Make BookText more of a dumb view. 
    //  With extra controller layer on top as a new class maybe.

    /**
     * @param {Object} config - Configuration object
     * @param {HTMLElement} config.textHolder - The container element for generated text blocks
     * @param {HTMLElement} config.fileNameLabel - The filename text, above the main text
     * @param {Function} config.onSeek - Callback when seeking audio(time) void
     * @param {Function} config.onPlay - Callback when play is requested () => Promise<void>
     * @param {Function} config.onStorePosition - Callback to store playback position (time) void
     * @param {Function} config.onShouldUpdateHighlight - Guard callback () => boolean
     * @param {Function} config.onAdvancedOnce - Callback when first segment advance occurs () void
     * @param {Function} config.onIsPaused - Callback to check if audio is paused () => boolean
     * @param {Function} config.onIsEnded - Callback to check if audio has ended () => boolean
     * @param {Function} config.onPause - Callback to pause audio () => void
     */
    constructor(config) {
        const requiredKeys = ["textHolder", "fileNameLabel", "onSeek", "onPlay", "onStorePosition", "onShouldUpdateHighlight", "onAdvancedOnce", "onIsPaused", "onIsEnded", "onPause"]
        const err = Util.validateObject(config, requiredKeys);
        if (err) {
            throw new Error(err)
        }

        this.textHolder = config.textHolder;
        this.fileNameLabel = config.fileNameLabel;

        // Event listeners
        this.textHolder.addEventListener('click', this.handleTextClick.bind(this));

        // Callbacks (functions provided by App)
        this.onSeek = config.onSeek;
        this.onPlay = config.onPlay;
        this.onStorePosition = config.onStorePosition;
        this.onShouldUpdateHighlight = config.onShouldUpdateHighlight;
        this.onAdvancedOnce = config.onAdvancedOnce;
        this.onIsPaused = config.onIsPaused;
        this.onIsEnded = config.onIsEnded;
        this.onPause = config.onPause;

        // Internal state
        this.textSegments = [];
        this.textSegmentGroups = [];
        this.audioIndices = []; // text segment indices that "have audio"
        this.isAudioTimelineOrdered = true;
        this.segmentMap = {}; // key = textSegment index, value = audioIndicies index or null -- ONE MORE level of indirection
        
        this.spans = [];
        this.segmentGroupSpans = [];
        this.currentIndex = -1; // the current segment index
        this.directSelections = [];
        this.hasAdvancedOnce = false;
        this.lastSeekTime = 0.0; // used for debouncing
    }

    /**
     * Initialize the controller with text segments.
     * 
     * `textSegments` is the normalized flat leaf sequence. `textSegmentGroups`
     * contains half-open ranges for ABR entries that were explicitly nested.
     * time_start and time_end are monotonically increasing,
     * with the exception that they can both be 0.0.
     */
    init(textSegments, addSectionDividers = false, sections = [], textSegmentGroups = []) {
        
        this.textSegments = textSegments;
        this.textSegmentGroups = Array.isArray(textSegmentGroups) ? textSegmentGroups : [];
        this.currentIndex = -1;
        this.directSelections = [];
        this.hasAdvancedOnce = false;

        // Init 'derived' data structures
        this.audioIndices = [];
        this.isAudioTimelineOrdered = true;
        let previousAudioEnd = -Infinity;
        for (const [i, segment] of this.textSegments.entries()) {
            if (BookText._isPlayableSegment(segment)) {
                this.audioIndices.push(i)
                if (segment.time_start < previousAudioEnd) {
                    this.isAudioTimelineOrdered = false;
                }
                previousAudioEnd = Math.max(previousAudioEnd, segment.time_end);
            }
        }
        this.segmentMap = {}
        for (const [i, audioIndex] of this.audioIndices.entries()) {
            this.segmentMap[audioIndex] = i
        }
        let lastIndex = -1
        for (let i = 0; i < this.textSegments.length; i++) { // Fill in undefined entries
            if (this.segmentMap[i] === undefined) {
                this.segmentMap[i] = lastIndex
            } else {
                lastIndex = this.segmentMap[i]
            }
        }

        this._populateText(addSectionDividers, sections);
    }

    /**
     * Clear all text and reset state
     */
    clear() {
        this.textHolder.innerHTML = "";
        this.textHolder.style.display = "none";
        this.spans = [];
        this.segmentGroupSpans = [];
        this.textSegments = [];
        this.textSegmentGroups = [];
        this.audioIndices = [];
        this.isAudioTimelineOrdered = true;
        this.segmentMap = {};
        this.currentIndex = -1;
        this.directSelections = [];
    }

    /**
     * Get the current active segment index
     * @returns {number} Current index or -1 if none
     */
    getCurrentIndex() {
        return this.currentIndex;
    }

    /**
     * Get span element by index
     * @param {number} i - Segment index
     * @returns {HTMLElement|null} The span element or null
     */
    getSpanByIndex(i) {
        return i >= 0 ? this.spans[i] : null;
    }

    /**
     * Get the currently highlighted span
     * @returns {HTMLElement|null} The current span or null
     */
    getCurrentSpan() {
        return this.currentIndex >= 0 ? this.spans[this.currentIndex] : null;
    }

    /**
     * Find the segment index for a given time position
     * @param {number} seconds - Time in seconds
     * @returns {number} Segment index or -1 if not found
     */
    getSegmentIndexBySeconds(seconds) {
        if (this.audioIndices.length === 0) {
            return -1;
        }

        // The overwhelmingly common poll-loop case needs no search.
        if (this.currentIndex >= 0) {
            const current = this.textSegments[this.currentIndex];
            if (seconds >= current.time_start && seconds < current.time_end) {
                return this.currentIndex;
            }
        }

        // Valid producer timelines are ordered and use the fast binary path.
        // Preserve permissive legacy behavior for unusual overlapping or
        // non-monotonic metadata rather than returning a misleading result.
        if (!this.isAudioTimelineOrdered) {
            for (const segmentIndex of this.audioIndices) {
                const segment = this.textSegments[segmentIndex];
                if (seconds >= segment.time_start && seconds < segment.time_end) {
                    return segmentIndex;
                }
            }
            return -1;
        }

        let lo = 0;
        let hi = this.audioIndices.length - 1;
        while (lo <= hi) {
            const mid = Math.floor((lo + hi) / 2);
            const segmentIndex = this.audioIndices[mid];
            const segment = this.textSegments[segmentIndex];

            if (seconds < segment.time_start) {
                hi = mid - 1;
            } else if (seconds >= segment.time_end) {
                lo = mid + 1;
            } else {
                return segmentIndex;
            }
        }

        return -1;
    }

    /**
     * Seeks to next/previous segment with audio using currentIndex if exists, else using timestamp
     */
    seekAdjacent(currentTime, isForward) {
        if (this.currentIndex > -1) {
            this.seekAdjacentFromIndex(this.currentIndex, isForward)
        } else {
            this.seekAdjacentFromTime(currentTime, isForward)
        }
    }

    /**
     * 
     */
    seekAdjacentFromIndex(segmentIndex, isForward) {
        const audioIndex = this.segmentMap[segmentIndex];
        const newAudioIndex = audioIndex + (isForward ? 1 : -1);
        if (newAudioIndex < 0 || newAudioIndex > this.audioIndices.length -1 ) {
            return;
        }
        const newSegmentIndex = this.audioIndices[newAudioIndex];
        this.seekBySegmentIndex(newSegmentIndex);
    }

    seekAdjacentFromTime(time, isForward) {
        const audioIndices = this.findAudioIndex(time)
        let newAudioIndex;
        if (audioIndices.length == 1) {
            newAudioIndex = audioIndices[0] + (isForward ? +1 : -1)
        } else { // array is two elements - previous and next
            newAudioIndex = isForward ? audioIndices[1] : audioIndices[0]
        }
        if (Number.isNaN(newAudioIndex) || newAudioIndex < 0 || newAudioIndex > this.audioIndices.length - 1) {
            return
        }
        const newSegmentIndex = this.audioIndices[newAudioIndex];
        this.seekBySegmentIndex(newSegmentIndex)
    }

    /**
     * Returns either a one-element array with the segment index containing the target time,
     * or a two-element array of the nearest indices (that have audio) "enclosing" the target time.
     */
    findAudioIndex(targetTime) {

        if (this.textSegments.length == 0 || this.audioIndices.length == 0) {
            return [NaN, NaN];
        }

        // Calc starting point for binary search
        const n = this.audioIndices.length;

        // Helper to get textSegment for an audioIndices entry
        const getSegment = (ai) => this.textSegments[this.audioIndices[ai]];

        // Edge cases
        if (targetTime < getSegment(0)["time_start"]) {
            return [NaN, 0];
        }
        if (targetTime > getSegment(n - 1)["time_end"]) {
            return [n - 1, NaN];
        }

        // Binary search on audioIndices to find segment enclosing targetTime
        let iters = -1; // for debugging
        let lo = 0;
        let hi = n - 1;

        while (lo <= hi) {
            
            iters++;
            
            const mid = Math.floor((lo + hi) / 2);
            const seg = getSegment(mid);

            if (targetTime >= seg["time_start"] && targetTime <= seg["time_end"]) {
                return [mid];
            }

            if (targetTime < seg["time_start"]) {
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }

        // targetTime falls between segments at lo-1 and lo
        return [lo - 1, lo];
    }

    /**
     * Seek to a specific segment by index
     * @param {number} i - Segment index
     * @param {boolean} andPlay - Whether to start playback after seeking
     * @returns {boolean} True if the seek was performed, false if debounced
     */
    seekBySegmentIndex(i, andPlay = false) {
        
        // Prevent frequent seeks bc looks glitchy, not ideal but
        if (Date.now() - this.lastSeekTime < 200) {
            return false;
        }
        this.lastSeekTime = Date.now()

        // Clear the old highlight while the poll loop catches up to the new
        // position -- unless seeking within the current segment, in which case
        // the poll will not re-apply it (no index change), so it is kept.
        if (i != this.currentIndex) {
            this.unhighlightByIndex(this.currentIndex);
        }
        const targetTime = this.textSegments[i]["time_start"];
        this.onSeek(targetTime);
        if (this.onIsPaused() && andPlay) {
            this._play();
        }
        return true;
    }

    /**
     * Smoothly scroll the text so that the given segment is brought into view.
     * 
     * Initiated directly (e.g. after a bookmark selection) rather than waiting
     * for the poll-driven highlight follow-along scrolling, which only fires
     * when the highlighted segment changes.
     * 
     * @param {number} i - Segment index
     */
    scrollToIndex(i) {
        this._scrollSpanIntoView(this.getSpanByIndex(i));
    }

    /**
     * Update highlight based on current audio time
     * Called from the app's poll loop
     * @param {number} currentTime - Current audio time in seconds
     * @returns {boolean} True if highlight changed, false otherwise
     */
    updateHighlight(currentTime) {
        if (!this.onShouldUpdateHighlight()) {
            return false;
        }

        const previousIndex = this.currentIndex;
        this.currentIndex = this.getSegmentIndexBySeconds(currentTime);

        if (this.currentIndex == previousIndex) {
            return false;
        }

        this.unhighlightByIndex(previousIndex);

        // Check if we've advanced naturally
        if (this.currentIndex - previousIndex == 1) {
            if (!this.onIsEnded()) {
                if (this.onStorePosition) {
                    this.onStorePosition(currentTime);
                }
            }
            if (!this.hasAdvancedOnce) {
                this.hasAdvancedOnce = false;
                this.onAdvancedOnce()
            }
        }

        if (this.currentIndex >= 0) {
            this._highlightSpan(this.currentIndex);
            this._scrollCurrentSpanIntoView();
        }

        return this.currentIndex != previousIndex;
    }

    /**
     * Handle text click events
     * @param {Event} event - Click event
     * @returns {number} Segment index clicked, or -1 if invalid
     */
    handleTextClick(event) {

        const target = event.target;
        const clickedSpan = target instanceof Element
            ? target.closest(".textSegment[data-segment-index]")
            : null;
        if (!clickedSpan || !this.textHolder.contains(clickedSpan)) {
            return -1;
        }

        const segmentIndex = Number(clickedSpan.dataset.segmentIndex);
        const segment = this.textSegments[segmentIndex];

        if (!segment || !BookText._isPlayableSegment(segment)) {
            return -1;
        }

        if (clickedSpan == this.getCurrentSpan()) {
            // Toggle play/pause on current span
            if (this.onIsPaused()) {
                this._play();
            } else {
                this.onPause();
            }
        } else {
            this.seekBySegmentIndex(segmentIndex, true);

            // Track direct selections
            const isSameAsLast = (this.directSelections.at(-1) !== undefined) && (this.directSelections.at(-1) == segmentIndex);
            if (!isSameAsLast) {
                this.directSelections.push(segmentIndex);
            }
        }

        return segmentIndex;
    }

    /**
     * Add bookmark CSS classes to spans
     * @param {Array} bookmarkIndices - Array of segment indices that are bookmarked
     */
    addBookmarkClasses(bookmarkIndices) {
        for (const index of bookmarkIndices) {
            if (this.spans[index]) {
                this.spans[index].classList.add("bookmark");
            }
        }
    }

    /**
     * Remove bookmark CSS classes from spans
     * @param {Array} bookmarkIndices - Array of segment indices to remove bookmark from
     */
    removeBookmarkClasses(bookmarkIndices) {
        for (const index of bookmarkIndices) {
            if (this.spans[index]) {
                this.spans[index].classList.remove("bookmark");
            }
        }
    }

    /**
     * Remove the leaf and containing-group highlights for an index.
     * @param {number} i - Flat leaf index
     */
    unhighlightByIndex(i) {
        if (i < 0) {
            return;
        }
        if (this.spans[i]) {
            this.spans[i].classList.remove("highlight");
        }
        if (this.segmentGroupSpans[i]) {
            this.segmentGroupSpans[i].classList.remove("groupHighlight");
        }
    }

    showFileName(fileName) {
        this.fileNameLabel.style.display = "block";
        this.fileNameLabel.textContent = fileName;
    }

    hideFileName() {
        this.fileNameLabel.style.display = "none";
    }

    // ========================================
    // Private Methods
    // ========================================

    /**
     * Build DOM from normalized flat leaves and nested-entry group ranges.
     * @param {boolean} addSectionDividers - Whether to add section break markers
     * @param {Array} sections - Section ranges over the flat leaf sequence
     * @private
     */
    _populateText(addSectionDividers, sections = []) {

        const sectionRanges = this._getDisplaySectionRanges(sections);
        const groupsByStartIndex = new Map();
        for (const group of this.textSegmentGroups) {
            groupsByStartIndex.set(group.startIndex, group);
        }

        this.spans = new Array(this.textSegments.length);
        this.segmentGroupSpans = new Array(this.textSegments.length).fill(null);

        const fragment = document.createDocumentFragment();
        for (const [sectionIndex, section] of sectionRanges.entries()) {
            const textBlock = document.createElement("pre");
            textBlock.className = "textHolder";
            textBlock.dataset.sectionIndex = sectionIndex;

            let i = section.startIndex;
            while (i < section.endIndex) {
                const group = groupsByStartIndex.get(i);
                if (group && group.endIndex <= section.endIndex) {
                    this._appendSegmentGroup(textBlock, group, addSectionDividers);
                    i = group.endIndex;
                } else {
                    this._appendFlatSegment(textBlock, i, addSectionDividers);
                    i += 1;
                }
            }

            fragment.appendChild(textBlock);
        }

        this.textHolder.replaceChildren(fragment);
        this.textHolder.classList.toggle("multipleTextHolders", sectionRanges.length > 1);
        this.textHolder.style.display = "block";
        for (const textBlock of this.textHolder.querySelectorAll(".textHolder")) {
            textBlock.style.display = "block";
        }
    }

    _appendFlatSegment(textBlock, index, addSectionDividers) {
        const split = BookText._splitTextSegment(this.textSegments[index].text);
        this._appendText(textBlock, split.before);
        textBlock.appendChild(this._makeSegmentSpan(index, split.content));
        this._appendTrailingText(textBlock, split.after, addSectionDividers);
    }

    _appendSegmentGroup(textBlock, group, addSectionDividers) {
        const firstIndex = group.startIndex;
        const lastIndex = group.endIndex - 1;
        const splits = [];
        for (let i = firstIndex; i <= lastIndex; i++) {
            splits.push(BookText._splitTextSegment(this.textSegments[i].text));
        }

        // Keep surrounding structural whitespace outside the inline group. All
        // separators between child leaves remain inside and receive group tint.
        this._appendText(textBlock, splits[0].before);

        const groupSpan = document.createElement("span");
        const groupHasAudio = this.textSegments
            .slice(firstIndex, lastIndex + 1)
            .some(BookText._isPlayableSegment);
        groupSpan.className = `segmentGroup ${groupHasAudio ? "hasAudio" : "noAudio"}`;
        groupSpan.dataset.groupIndex = group.groupIndex;
        groupSpan.dataset.groupStart = group.startIndex;
        groupSpan.dataset.groupEnd = group.endIndex;
        groupSpan.dataset.leafCount = group.leafCount;

        for (let i = firstIndex; i <= lastIndex; i++) {
            const split = splits[i - firstIndex];
            if (i > firstIndex) {
                this._appendText(groupSpan, split.before);
            }

            groupSpan.appendChild(this._makeSegmentSpan(i, split.content));
            this.segmentGroupSpans[i] = groupSpan;

            if (i < lastIndex) {
                // Section-divider markup is block content and must not be put
                // inside an inline group. Valid producer data does not place a
                // structural divider between children of one generated phrase.
                this._appendText(groupSpan, split.after);
            }
        }

        textBlock.appendChild(groupSpan);
        this._appendTrailingText(textBlock, splits.at(-1).after, addSectionDividers);
    }

    _makeSegmentSpan(index, content) {
        const segment = this.textSegments[index];
        const span = document.createElement("span");
        span.id = `segment-${index}`;
        span.className = `textSegment ${BookText._isPlayableSegment(segment) ? "hasAudio" : "noAudio"}`;
        span.dataset.segmentIndex = index;
        span.textContent = content;
        this.spans[index] = span;
        return span;
    }

    _appendText(parent, text) {
        if (text) {
            parent.appendChild(document.createTextNode(text));
        }
    }

    _appendTrailingText(parent, text, addSectionDividers) {
        if (!text) {
            return;
        }

        const numLfs = text.split('\n').length - 1;
        if (!addSectionDividers || numLfs < 3) {
            this._appendText(parent, text);
            return;
        }

        parent.appendChild(document.createElement("br"));
        parent.appendChild(document.createTextNode("\u00a0"));
        parent.appendChild(document.createElement("hr"));
        parent.appendChild(document.createElement("br"));
    }

    _getDisplaySectionRanges(sections) {
        if (!Array.isArray(sections) || sections.length === 0) {
            return [{ startIndex: 0, endIndex: this.textSegments.length }];
        }

        const ranges = [];
        let expectedStartIndex = 0;
        for (const section of sections) {
            const startIndex = section.start_index;
            const endIndex = section.end_index;
            if (!Number.isInteger(startIndex) || !Number.isInteger(endIndex)) {
                return [{ startIndex: 0, endIndex: this.textSegments.length }];
            }
            if (startIndex < 0 || endIndex <= startIndex || endIndex > this.textSegments.length) {
                return [{ startIndex: 0, endIndex: this.textSegments.length }];
            }
            if (startIndex !== expectedStartIndex) {
                return [{ startIndex: 0, endIndex: this.textSegments.length }];
            }
            ranges.push({ startIndex, endIndex });
            expectedStartIndex = endIndex;
        }

        if (ranges.length === 0 || expectedStartIndex !== this.textSegments.length) {
            return [{ startIndex: 0, endIndex: this.textSegments.length }];
        }

        // One inline group cannot span sibling <pre> text blocks. Producer data
        // aligns these boundaries, but malformed/third-party metadata should
        // still preserve group rendering by falling back to one text block.
        const sectionBoundaries = ranges.slice(0, -1).map((range) => range.endIndex);
        let boundaryIndex = 0;
        for (const group of this.textSegmentGroups) {
            while (
                boundaryIndex < sectionBoundaries.length
                && sectionBoundaries[boundaryIndex] <= group.startIndex
            ) {
                boundaryIndex += 1;
            }
            if (
                boundaryIndex < sectionBoundaries.length
                && sectionBoundaries[boundaryIndex] < group.endIndex
            ) {
                return [{ startIndex: 0, endIndex: this.textSegments.length }];
            }
        }

        return ranges;
    }

    /**
     * Highlight a specific span
     * @param {number} index - Segment index to highlight
     * @private
     */
    _highlightSpan(index) {
        if (index >= 0 && this.spans[index]) {
            this.spans[index].classList.add("highlight");
        }
        if (index >= 0 && this.segmentGroupSpans[index]) {
            this.segmentGroupSpans[index].classList.add("groupHighlight");
        }
    }

    /**
     * Scroll the current span into view
     * @private
     */
    _scrollCurrentSpanIntoView() {
        this._scrollSpanIntoView(this.getCurrentSpan());
    }

    /**
     * Smoothly scroll a span element into view
     * @param {HTMLElement|null} span - The span element, or null
     * @private
     */
    _scrollSpanIntoView(span) {
        if (span && !(document.activeElement instanceof HTMLInputElement)) {
            span.scrollIntoView({
                behavior: 'smooth',
                block: 'center',
                inline: 'nearest'
            });
        }
    }

    /**
     * Play audio using the callback if available
     * @private
     */
    async _play() {
        if (this.onPlay) {
            await this.onPlay();
        }
    }

    static _isPlayableSegment(segment) {
        if (!segment) {
            return false;
        }
        if (typeof segment.playable === "boolean") {
            return segment.playable;
        }
        return Number.isFinite(segment.time_start)
            && Number.isFinite(segment.time_end)
            && segment.time_end > segment.time_start;
    }

    static _splitTextSegment(text) {

        // Special case; not great
        let o = BookText._splitOrnamentalBreak(text)
        if (o["after"]) {
            return o
        }

        o = Util.splitWhitespace(text);
        return o;
    }

    /**
     * Splits string into two parts if it ends with so-called ornamental line break
     * (line feed followed by non-number/non-letter characters).
     * 
     * This is part of a workaround to account for special case where text segments 
     * end with a line feed followed by 'ornamental' characters.
     */
    static _splitOrnamentalBreak(text) {
        
        const regex = /[\r\n][^\p{L}\p{N}]*$/u; // unicode categories "L" and "N"
        const match = text.match(regex);
        if (match) {
            const splitIndex = text.length - match[0].length;
            return {
                "content": text.slice(0, splitIndex),
                "after": match[0]
            }
        } else {
            return {
                "content": text,
                "after": ""
            }
        }
    }

}
