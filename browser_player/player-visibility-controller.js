"use strict";

/**
 * Manages the visibility of the audio player based on mouse position.
 * Handles showing/hiding the player when the mouse enters/leaves the player area.
 * On touch devices or when the player is pinned, the player remains visible.
 */
class PlayerVisibilityController {

    /**
     * @param {Object} config - Configuration object
     * @param {HTMLElement} config.playerOverlay - The overlay element to check mouse position against
     * @param {HTMLElement} config.playerHolder - The player holder element to show/hide
     * @param {AudioPlayer} config.player - AudioPlayer instance to check pinned state
     * @param {Function} [config.onVisibilityChange] - Optional callback when visibility changes (isInPlayer) => void
     */
    constructor(config) {
        const err = Util.validateObject(config, ["playerOverlay", "playerHolder", "player"]);
        if (err) {
            throw new Error(err);
        }

        this.playerOverlay = config.playerOverlay;
        this.playerHolder = config.playerHolder;
        this.player = config.player;
        this.onVisibilityChange = config.onVisibilityChange || (() => {});

        // Internal state
        this.mousePosition = { x: -1, y: -1 };
        this.isInPlayer = false;
        this.isTouchDevice = Util.isTouchDevice();

        // Initialize listeners
        this._initListeners();
    }

    // ========================================
    // Public Methods
    // ========================================

    /**
     * Check and update player visibility based on mouse position.
     * Called from the poll loop.
     */
    updateVisibility() {
        if (this.isTouchDevice || this.player.isPinned()) {
            return;
        }

        const wasInPlayer = this.isInPlayer;
        this.isInPlayer = this.isMouseOverPlayer();

        if (this.isInPlayer === wasInPlayer) {
            return;
        }

        if (this.isInPlayer) {
            this.show();
        } else {
            this.hide();
        }

        this.onVisibilityChange(this.isInPlayer);
    }

    /**
     * Check if mouse is currently over the player overlay.
     * @returns {boolean}
     */
    isMouseOverPlayer() {
        const els = document.elementsFromPoint(this.mousePosition.x, this.mousePosition.y);
        return els.indexOf(this.playerOverlay) > -1;
    }

    /**
     * Show the player holder element.
     */
    show() {
        ShowUtil.show(this.playerHolder);
    }

    /**
     * Hide the player holder element.
     */
    hide() {
        ShowUtil.hide(this.playerHolder, false);
    }

    /**
     * Get the current isInPlayer state.
     * @returns {boolean}
     */
    isInPlayerState() {
        return this.isInPlayer;
    }

    // ========================================
    // Private Methods
    // ========================================

    /**
     * Initialize event listeners.
     * @private
     */
    _initListeners() {
        window.addEventListener('mousemove', this._onMouseMove.bind(this));
    }

    /**
     * Handle mouse move events.
     * @param {MouseEvent} event
     * @private
     */
    _onMouseMove(event) {
        this.mousePosition.x = event.clientX;
        this.mousePosition.y = event.clientY;
    }
}
