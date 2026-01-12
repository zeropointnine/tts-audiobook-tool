"use strict";

/**
 * Manages toast notifications with auto-hide functionality.
 * Plus specific use case "Click to play...".
 */
class Toast {

    /**
     * @param {Object} config - Configuration object
     * @param {HTMLElement} config.toast - The toast element
     * @param {Function} config.onPlay - Callback when play is requested () => Promise<void>
     */
    constructor(config) {
        const err = Util.validateObject(config, ["toast"], ["onPlay"]);
        if (err) {
            throw new Error(err);
        }

        this.toast = config.toast;
        this.onPlay = config.onPlay; // optional

        // Internal state
        this.isPlayPrompt = false;
        this.toastHideDelayId = -1;

        // Initialize listeners
        this._initListeners();
    }

    /**
     * Show a toast message
     * @param {string} message - The message to display
     * @param {boolean} isPlayPrompt - Whether this is a play-prompt toast (clickable to play)
     */
    show(message, isPlayPrompt = false) {
        clearTimeout(this.toastHideDelayId);
        this.isPlayPrompt = isPlayPrompt;
        this.toast.textContent = message;

        this._showElement();
        if (!isPlayPrompt) {
            this.toastHideDelayId = setTimeout(() => { this.hide(); }, 2500);
        }
    }

    /**
     * Hide the toast
     */
    hide() {
        clearTimeout(this.toastHideDelayId);
        this.isPlayPrompt = false;
        this._hideElement();
    }

    /**
     * Check if toast is currently showing as a play-prompt
     * @returns {boolean}
     */
    isShowingPlayPrompt() {
        return this.isPlayPrompt;
    }

    // ========================================
    // Private Methods
    // ========================================

    /**
     * Initialize event listeners
     * @private
     */
    _initListeners() {
        this.toast.addEventListener("click", this._onClick.bind(this));
    }

    /**
     * Handle toast click event
     * @private
     */
    _onClick() {
        if (this.isPlayPrompt) {
            this._playAudio();
        }
        this._hideElement();
    }

    /**
     * Play audio using the callback
     * @private
     */
    async _playAudio() {
        if (this.onPlay) {
            await this.onPlay();
        }
    }

    /**
     * Show the toast element
     * @private
     */
    _showElement() {
        this.toast.removeEventListener('transitionend', this._onHideElementDisplayNone);
        this.toast.style.display = "block";
        this.toast.offsetHeight; // Force reflow
        this.toast.classList.add("showing");
    }

    /**
     * Hide the toast element
     * @private
     */
    _hideElement() {
        this.toast.removeEventListener('transitionend', this._onHideElementDisplayNone);
        this.toast.addEventListener('transitionend', this._onHideElementDisplayNone, { once: true });
        this.toast.classList.remove("showing");
    }

    /**
     * Handle transition end to set display: none
     * @param {Event} e
     * @private
     */
    _onHideElementDisplayNone(e) {
        e.currentTarget.style.display = "none";
    }
}
