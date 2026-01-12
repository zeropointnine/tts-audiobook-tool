"use strict";

/**
 * Manages the sleep timer feature for auto-pausing audio playback.
 * Displays countdown and automatically pauses audio when timer expires.
 */
class SleepTimer {

    /**
     * @param {Object} config - Configuration object
     * @param {HTMLAudioElement} config.audio - The audio element to pause
     * @param {RootAttributer} config.rootAttributer
     * @param {HTMLElement} config.sleepTimeLeft - Element to display countdown
     * @param {Function} config.onShowToast - Callback to show toast messages (message) void
     * @param {Function} config.onUpdateMenu - Callback to update menu button states
     */
    constructor(config) {
        const requiredKeys = ["audio", "rootAttributer", "sleepTimeLeft", "onShowToast", "onUpdateMenu"]
        const err = Util.validateObject(config, requiredKeys);
        if (err) {
            throw new Error(err)
        }

        this.SLEEP_MS = (1000 * 60) * 15; // 15 minutes
 
        this.audio = config.audio;
        this.rootAttributer = config.rootAttributer;
        this.sleepTimeLeft = config.sleepTimeLeft;
        this.onShowToast = config.onShowToast;
        this.onUpdateMenu = config.onUpdateMenu;

        this.sleepIntervalId = -1;
        this.sleepEndTime = -1;
    }

    /**
     * Start the sleep timer (15 minutes until auto-pause)
     */
    start() {
        this.onShowToast("Sleep timer: Will auto-pause in 15 minutes");
        this.sleepEndTime = new Date().getTime() + this.SLEEP_MS;

        clearInterval(this.sleepIntervalId);
        this._onInterval();
        this.sleepIntervalId = setInterval(this._onInterval.bind(this), 1000);
    }

    /**
     * Cancel the sleep timer
     * @param {boolean} - True if timer finished naturally
     */
    clear(isFinishedMessage = false) {
        const message = isFinishedMessage ? "Sleep timer finished" : "Sleep timer cancelled";
        this.onShowToast(message);

        this.rootAttributer.set("data-sleep", "");
        clearInterval(this.sleepIntervalId);
        this.onUpdateMenu();
    }

    // ========================================
    // Private Methods
    // ========================================

    /**
     * Called each second to update countdown and check expiration
     * @private
     */
    _onInterval() {
        const ms = this.sleepEndTime - new Date().getTime();
        if (ms <= 0) {
            this.audio.pause();
            this.clear(true);
            return;
        }
        this.sleepTimeLeft.textContent = Util.msToString(ms);
    }
}