"use strict";

/**
 * Manages and displays the book text widget.
 * 
 * Uses multiple callbacks to interacts with <audio> element, etc.
 */
class BookText {

    /**
     * @param {Object} config - Configuration object
     * @param {HTMLElement} config.textHolder - The container element for text
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
        this.spans = [];
        this.currentIndex = -1;
        this.directSelections = [];
        this.hasAdvancedOnce = false;
    }

    /**
     * Initialize the controller with text segments.
     * Also sets the filename text
     * 
     * @param {Array} textSegments - Array of segment objects with text, time_start, time_end
     * @param {boolean} addSectionDividers - Whether to add visual dividers for section breaks
     */
    init(textSegments, addSectionDividers = false) {
        this.textSegments = textSegments;
        this.currentIndex = -1;
        this.directSelections = [];
        this.hasAdvancedOnce = false;
        this._populateText(addSectionDividers);
    }

    /**
     * Clear all text and reset state
     */
    clear() {
        this.textHolder.innerHTML = "";
        this.textHolder.style.display = "none";
        this.spans = [];
        this.textSegments = [];
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
        if (this.textSegments.length == 0) {
            return -1;
        }

        const baseIndex = Math.max(this.currentIndex, 0);

        let delta = 0;
        while (true) {
            const indexInc = baseIndex + delta;
            const indexDec = baseIndex - delta;

            const oob = (indexInc >= this.textSegments.length && indexDec < 0);
            if (oob) {
                return -1;
            }

            if (indexDec >= 0) {
                const segment = this.textSegments[indexDec];
                if (seconds >= segment["time_start"] && seconds < segment["time_end"]) {
                    return indexDec;
                }
            }
            if (indexInc < this.textSegments.length) {
                const segment = this.textSegments[indexInc];
                if (seconds >= segment["time_start"] && seconds < segment["time_end"]) {
                    return indexInc;
                }
            }

            delta += 1;
        }
    }

    /**
     * Seek to the next playable segment
     */
    seekNextSegment() {
        let index = this.currentIndex;
        for (let i = index + 1; i <= index + 100; i++) {
            if (i >= this.textSegments.length) {
                return;
            }
            const segment = this.textSegments[i];
            const has_time = (segment["time_end"] > 0);
            if (has_time) {
                this.seekBySegmentIndex(i);
                return;
            }
        }
    }

    /**
     * Seek to the previous playable segment
     */
    seekPreviousSegment() {
        if (this.currentIndex <= 0) {
            return;
        }
        for (let i = this.currentIndex - 1; i >= this.currentIndex - 100; i--) {
            if (i < 0) {
                return;
            }
            const segment = this.textSegments[i];
            const has_time = (segment["time_end"] > 0);
            if (has_time) {
                this.seekBySegmentIndex(i);
                return;
            }
        }
    }

    /**
     * Seek to a specific segment by index
     * @param {number} i - Segment index
     * @param {boolean} andPlay - Whether to start playback after seeking
     */
    seekBySegmentIndex(i, andPlay = false) {
        this.unhighlightByIndex(this.currentIndex);
        const targetTime = this.textSegments[i]["time_start"];
        this.onSeek(targetTime);
        if (this.onIsPaused() && andPlay) {
            this._play();
        }
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

        const clickedSpan = event.target;

        const isSegment = (clickedSpan.tagName === 'SPAN' && clickedSpan.id.startsWith("segment-"));
        if (!isSegment) {
            return -1;
        }

        const segmentIndex = parseInt(clickedSpan.id.split("-")[1]);
        const segment = this.textSegments[segmentIndex];

        if (segment["time_start"] == 0 && segment["time_end"] == 0) {
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
     * Remove highlight from a range of spans around the given index
     * @param {number} i - Center index
     */
    unhighlightByIndex(i) {
        if (i < 0) {
            return;
        }
        const a = Math.max(i - 20, 0);
        const b = Math.min(i + 20, this.textSegments.length - 1);
        for (let j = a; j <= b; j++) {
            if (this.spans[j]) {
                this.spans[j].classList.remove("highlight");
            }
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
     * Build DOM from text segments
     * @param {boolean} addSectionDividers - Whether to add section break markers
     * @private
     */
    _populateText(addSectionDividers) {
        let contentHtml = '';
        this.textSegments.forEach((segment, i) => {
            const o = Util.splitWhitespace(segment.text);

            if (o["before"]) {
                contentHtml += Util.escapeHtml(o["before"]);
            }

            const hasAudio = segment["time_start"] > 0.0 && segment["time_end"] > 0.0;
            const className = hasAudio ? "hasAudio" : "noAudio";
            const spanString = `<span id="segment-${i}" class="${className}">${Util.escapeHtml(o["content"])}</span>`;
            contentHtml += spanString;

            if (o["after"]) {
                if (addSectionDividers) {
                    const numLfs = o["after"].split('\n').length - 1;
                    if (numLfs >= 3) {
                        contentHtml += "<br>&nbsp;<hr><br>";
                    } else {
                        contentHtml += Util.escapeHtml(o["after"]);
                    }
                } else {
                    contentHtml += Util.escapeHtml(o["after"]);
                }
            }
        });

        this.textHolder.innerHTML = contentHtml;
        this.textHolder.style.display = "block";

        this.spans = [];
        for (let i = 0; i < this.textSegments.length; i++) {
            this.spans[i] = document.getElementById("segment-" + i);
        }
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
    }

    /**
     * Scroll the current span into view
     * @private
     */
    _scrollCurrentSpanIntoView() {
        const currentSpan = this.getCurrentSpan();
        if (currentSpan && !(document.activeElement instanceof HTMLInputElement)) {
            currentSpan.scrollIntoView({
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
}