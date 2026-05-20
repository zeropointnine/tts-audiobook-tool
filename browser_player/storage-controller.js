"use strict";

/**
 * Manages all localStorage operations for the application.
 * Handles playback position, bookmarks, settings, and app state persistence.
 */
class StorageController {

    /**
     * @param {Object} config - Configuration object
     * @param {string} config.positionId - Timeline identifier for currentTime resume
     * @param {string} config.bookmarkId - Text-structure identifier for bookmark indices
     */
    constructor(config = {}) {
        this.positionId = config.positionId || null;
        this.bookmarkId = config.bookmarkId || null;
    }

    // ========================================
    // Identity Management
    // ========================================

    /**
     * Set the current IDs for storage operations.
     * @param {Object} identity - Identity object
     * @param {string} identity.positionId - Timeline identifier for position storage
     * @param {string} identity.bookmarkId - Text identifier for bookmark storage
     */
    setIdentity(identity) {
        this.positionId = identity.positionId || null;
        this.bookmarkId = identity.bookmarkId || null;
    }

    /**
     * Set both storage identities from one value.
     * @param {string} fileId - The shared storage identifier
     */
    setFileId(fileId) {
        this.setIdentity({
            positionId: fileId,
            bookmarkId: fileId,
        });
    }

    /**
     * Get the current shared ID, if position/bookmark IDs are the same.
     * @returns {string|null} The current shared ID
     */
    getFileId() {
        return (this.positionId === this.bookmarkId) ? this.positionId : null;
    }

    // ========================================
    // Position Storage
    // ========================================

    /**
     * Store the current playback position for the current file
     * @param {number} value - The position in seconds
     */
    storePosition(value) {
        if (!this.positionId) {
            return;
        }
        const key = this._getPositionKey();
        localStorage.setItem(key, value);
    }

    /**
     * Load the stored playback position for the current file
     * @returns {number|null} The stored position in seconds, or null if not found
     */
    loadPosition() {
        if (!this.positionId) {
            return null;
        }
        const value = localStorage.getItem(this._getPositionKey());
        if (!value) {
            return null;
        }
        const parsed = parseFloat(value);
        return isNaN(parsed) ? null : parsed;
    }

    /**
     * Clear the stored position for the current file
     */
    clearPosition() {
        if (!this.positionId) {
            return;
        }
        const key = this._getPositionKey();
        localStorage.removeItem(key);
    }

    // ========================================
    // Bookmark Storage
    // ========================================

    /**
     * Load bookmarks for the current file
     * @returns {Array} Array of bookmark indices
     */
    loadBookmarks() {
        if (!this.bookmarkId) {
            return [];
        }
        const value = localStorage.getItem(this._getBookmarksKey());
        if (!value) {
            return [];
        }
        let arr = [];
        try {
            arr = JSON.parse(value);
        } catch (exc) {
            console.log("json exception:", value);
            return [];
        }
        if (!Array.isArray(arr)) {
            console.log("not an array:", arr);
            return [];
        }
        return arr;
    }

    /**
     * Save bookmarks for the current file
     * @param {Array} indices - Array of bookmark indices
     */
    saveBookmarks(indices) {
        if (!this.bookmarkId) {
            return;
        }
        const key = this._getBookmarksKey();
        const value = JSON.stringify(indices);
        localStorage.setItem(key, value);
    }

    /**
     * Check if bookmarks exist for the current file
     * @returns {boolean} True if bookmarks exist
     */
    hasBookmarks() {
        if (!this.bookmarkId) {
            return false;
        }
        return localStorage.getItem(this._getBookmarksKey()) !== null;
    }

    // ========================================
    // Settings Persistence
    // ========================================

    /**
     * Load all settings that start with "data-" prefix
     * @returns {Object} Object with key-value pairs
     */
    loadDataAttributes() {
        const settings = {};
        const keys = Object.keys(localStorage);
        keys.forEach(key => {
            if (key.startsWith("data-")) {
                settings[key] = localStorage[key];
            }
        });
        return settings;
    }

    /**
     * Save a setting value
     * @param {string} key - The setting key
     * @param {string} value - The setting value
     */
    saveSetting(key, value) {
        localStorage.setItem(key, value);
    }

    /**
     * Load a setting value
     * @param {string} key - The setting key
     * @returns {string|null} The setting value or null if not found
     */
    loadSetting(key) {
        return localStorage.getItem(key);
    }

    /**
     * Remove a setting
     * @param {string} key - The setting key
     */
    removeSetting(key) {
        localStorage.removeItem(key);
    }

    // ========================================
    // App State
    // ========================================

    /**
     * Load the last opened file name
     * @returns {string|null} The last opened file name or null
     */
    loadLastOpened() {
        return localStorage.getItem("last_opened");
    }

    /**
     * Save the last opened file name
     * @param {string} name - The file name
     */
    saveLastOpened(name) {
        localStorage.setItem("last_opened", name);
    }

    /**
     * Clear the last opened file name
     */
    clearLastOpened() {
        localStorage.setItem("last_opened", "");
    }

    // ========================================
    // Player Preferences
    // ========================================

    /**
     * Load the player pinned preference
     * @returns {boolean} True if player should be pinned
     */
    loadPlayerPinned() {
        const s = localStorage.getItem("pref_player_pinned");
        return (s === "1") || !s;
    }

    /**
     * Save the player pinned preference
     * @param {boolean} isPinned - Whether the player is pinned
     */
    savePlayerPinned(isPinned) {
        localStorage.setItem("pref_player_pinned", isPinned ? "1" : "0");
    }

    /**
     * Load the player volume preference
     * @returns {number|null} Volume in range [0, 1], or null if not found/invalid
     */
    loadVolume() {
        const s = localStorage.getItem("pref_volume");
        if (s === null) {
            return null;
        }
        const value = parseFloat(s);
        if (isNaN(value)) {
            return null;
        }
        return Math.max(0, Math.min(1, value));
    }

    /**
     * Save the player volume preference
     * @param {number} volume - Volume in range [0, 1]
     */
    saveVolume(volume) {
        const clamped = Math.max(0, Math.min(1, volume));
        localStorage.setItem("pref_volume", clamped.toString());
    }

    // ========================================
    // Private Methods
    // ========================================

    /**
     * Get the localStorage key for position storage.
     * @private
     * @returns {string} The position key
     */
    _getPositionKey() {
        return `abrPlayer:v3:position:${this.positionId}`;
    }

    /**
     * Get the localStorage key for bookmark storage.
     * @private
     * @returns {string} The bookmarks key
     */
    _getBookmarksKey() {
        return `abrPlayer:v3:bookmarks:${this.bookmarkId}`;
    }
}
