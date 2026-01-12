"use strict";

/**
 * Menu panel widget (vertical column of buttons).
 * 
 * (Menu toggle button handled elsewhere)
 */
class Menu {

    /**
     * @param {Object} config - Configuration object
     * @param {RootAttributer} config.rootAttributer
     * @param {StorageController} config.storageController - Storage controller for persistence
     * @param {HTMLElement} config.mainEl - The menu panel element
     * @param {Function} config.onScrollTop - Callback to scroll to top () => void
     * @param {Function} config.onHidePanels - Callback to hide scrim and panels () => void
     * @param {Function} config.onSleepTimerStart - Callback to start the sleep timer () => void
     * @param {Function} config.onSleepTimerClear - Callback to clear the sleep timer () => void
     */
    constructor(config) {
        
        const requiredKeys = ["mainEl", "rootAttributer", "storageController", "onScrollTop", "onHidePanels", "onSleepTimerStart", "onSleepTimerClear"]
        const err = Util.validateObject(config, requiredKeys);
        if (err) {
            throw new Error(err)
        }

        this.mainEl = config.mainEl;

        this.scrollTopButton = this.mainEl.querySelector("#scrollTopButton");
        this.textSizeButton = this.mainEl.querySelector("#textSizeButton");
        this.themeButton = this.mainEl.querySelector("#themeButton");
        this.colorsButton = this.mainEl.querySelector("#segmentColorsButton");
        this.sleepButton = this.mainEl.querySelector("#sleepButton");

        this.rootAttributer = config.rootAttributer;
        this.storageController = config.storageController;

        const els = [this.scrollTopButton, this.textSizeButton, this.themeButton, this.colorsButton, this.sleepButton];
        for (const el of els) {
            if (!el) {
                throw new Error("Missing html element")
            }
        }

        this.onScrollTop = config.onScrollTop;
        this.onHidePanels = config.onHidePanels;
        this.onSleepTimerStart = config.onSleepTimerStart;
        this.onSleepTimerClear = config.onSleepTimerClear;

        // Initialize click listeners
        this._initClickListeners();
    }

    /**
     * Show the menu panel element
     */
    show() {
       ShowUtil.show(this.mainEl, "flex");
    }

    /**
     * Hide the menu panel element
     */
    hide() {
       ShowUtil.hide(this.mainEl);
    }

    get isShowing() {
        return ShowUtil.isShowing(this.mainEl);
    }

    /**
     * Update menu button state based on current settings
     */
    update() {
        this.setButtonChildVisible(this.textSizeButton, this.rootAttributer.get("data-book-text-size"));
        this.setButtonChildVisible(this.themeButton, this.rootAttributer.get("data-theme"));
        this.setButtonChildVisible(this.sleepButton, this.rootAttributer.get("data-sleep"));
        // this.colorsButton has one child which always remains visible
    }

    /**
     * Show only the button's child element whose data-value matches the target value
     * @param {HTMLElement} holder - Parent element containing children with data-value attributes
     * @param {string} targetValue - The value to match
     */
    setButtonChildVisible(holder, targetValue) {

        for (const child of holder.children) {
            child.style.display = "none";
        }

        for (const child of holder.children) {
            const value = child.dataset["value"];

            let isMatch = (value === targetValue);
            isMatch |= (!value || value == "default") && (!targetValue || targetValue == "default");
            if (isMatch) {
                child.style.display = "revert";
                break;
            }
        }
    }

    /**
     * Apply settings from storage to the root element
     * @param {Object} settings - Key-value pairs of settings to apply
     */
    applySettings(settings) {
        for (const [key, value] of Object.entries(settings)) {
            if (value === "" || value === null || value === undefined) {
                this.rootAttributer.set(key, null);
            } else {
                this.rootAttributer.set(key, value);
            }
        }
    }

    /**
     * Get the current value of a setting
     * @param {string} attribute - The attribute name
     * @returns {string} The current value (empty string if not set)
     */
    getSetting(attribute) {
        return this.rootAttributer.get(attribute);
    }

    /**
     * Set a setting value directly
     * @param {string} attribute - The attribute name
     * @param {string} value - The value to set
     * @param {boolean} persist - Whether to persist to storage
     */
    setSetting(attribute, value, persist = true) {
        if (value === "" || value === null || value === undefined) {
            this.rootAttributer.set(attribute, null);
        } else {
            this.rootAttributer.set(attribute, value);
        }

        if (persist) {
            this.storageController.saveSetting(attribute, value);
        }
    }

    // ========================================
    // Click Handler Initialization
    // ========================================

    /**
     * Initialize click listeners for menu panel buttons
     * @private
     */
    _initClickListeners() {
        this.mainEl.addEventListener("click", (e) => { e.stopPropagation(); });
        this.scrollTopButton.addEventListener('click', this._onScrollTopClick.bind(this));
        this.textSizeButton.addEventListener('click', this._onTextSizeButtonClick.bind(this));
        this.themeButton.addEventListener('click', this._onThemeButtonClick.bind(this));
        this.colorsButton.addEventListener('click', this._onSegmentColorsButtonClick.bind(this));
        this.sleepButton.addEventListener("click", this._onSleepButtonClick.bind(this));
    }

    /**
     * Handle scroll to top button click
     * @param {Event} e
     * @private
     */
    _onScrollTopClick(e) {
        e.stopPropagation();
        this.onHidePanels();
        this.onScrollTop();
    }

    /**
     * Handle text size button click
     * @param {Event} e
     * @private
     */
    _onTextSizeButtonClick(e) {
        e.stopPropagation();
        this.rootAttributer.cycle("data-book-text-size", ["medium", "small"]);
        this.update();
    }

    /**
     * Handle theme button click
     * @param {Event} e
     * @private
     */
    _onThemeButtonClick(e) {
        e.stopPropagation();
        this.rootAttributer.cycle("data-theme", ["dark"]);
        this.update();
    }

    /**
     * Handle segment colors button click
     * @param {Event} e
     * @private
     */
    _onSegmentColorsButtonClick(e) {
        e.stopPropagation();
        this.rootAttributer.cycle("data-segment-colors", ["blue", "red"]);
        this.update();
    }

    /**
     * Handle sleep button click
     * @param {Event} e
     * @private
     */
    _onSleepButtonClick(e) {
        e.stopPropagation();
        const value = this.rootAttributer.cycle("data-sleep", ["on"], false);
        if (value == "on") {
            this.onSleepTimerStart();
        } else {
            this.onSleepTimerClear();
        }
        this.update();
        this.onHidePanels();
    }
}
