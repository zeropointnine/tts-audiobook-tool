"use strict";

/**
 * Detects and handles "zombie" state, which is where the browser drops
 * the audio file handle while the tab is in the background.
 * 
 * This can happen on mobile when the tab is backgrounded (either 
 * because another tab is active or because the browser is backgrounded)
 * while the audio is paused.
 */
class ZombieChecker {

    /**
     * @param {HTMLAudioElement} audio - The audio element to test
     * @param {function} onDetected - Callback for when zombie state is detected
     */
    constructor(audio, onDetected) {
        if (!audio || !onDetected) {
            throw new Error("Missing argument");
        }
        this.audio = audio;
        this.onDetected = onDetected;

        this._isCheckingZombie = false;
    }

    /**
     * Get the current checking state
     * @returns {boolean} True if currently checking for zombie state
     */
    get isChecking() {
        return this._isCheckingZombie;
    }

    /**
     * Check if the audio file handle has been dropped by the browser.
     * This is typically called when the mobile browser tab becomes visible again.
     */
    check() {
        if (this.isChecking) {
            return;
        }
        if (!(this.audio.duration > 0) || !this.audio.paused || this.audio.ended) {
            return;
        }

        this._isCheckingZombie = true;
        let timeoutId = 0;
        const originalCurrentTime = this.audio.currentTime;
        const originalVolume = this.audio.volume;
        let timeUpdateCount = 0;

        const cleanUp = () => {
            this.audio.removeEventListener("timeupdate", onTimeUpdate);
            this.audio.removeEventListener("ended", onEnded);
            clearTimeout(timeoutId);

            this._isCheckingZombie = false;
            this.audio.pause();
            this.audio.volume = originalVolume;
        };

        const doFailed = () => {
            cleanUp();
            this.onDetected(originalCurrentTime);
        };

        const onEnded = () => {
            doFailed();
        };

        const onTimeUpdate = () => {
            if (this.audio.currentTime >= this.audio.duration) {
                // The playhead has "jumped" to the end == must be a fail
                doFailed();
                return;
            }
            timeUpdateCount += 1;
            if (timeUpdateCount >= 3) {
                cleanUp();
            }
        };

        const onTimeout = () => {
            // Timed out. "Indeterminate".
            cleanUp();
        };

        this.audio.addEventListener("timeupdate", onTimeUpdate);
        this.audio.addEventListener("ended", onEnded, { once: true });
        timeoutId = setTimeout(onTimeout, 1000);
        this.audio.volume = 0.0;
        this.audio.play();
    }
}
