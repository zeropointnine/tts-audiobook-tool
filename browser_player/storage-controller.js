"use strict";

/**
 * Manages all localStorage operations for the application.
 * Handles playback position, bookmarks, settings, and app state persistence.
 */
class StorageController {

    /**
     * @param {Object} config - Configuration object
     * @param {string} config.fileId - The current file identifier (can be updated later)
     */
    constructor(config = {}) {
        this.fileId = config.fileId || null;
    }

    // ========================================
    // File ID Management
    // ========================================

    /**
     * Set the current file ID for storage operations
     * @param {string} fileId - The file identifier
     */
    setFileId(fileId) {
        this.fileId = fileId;
    }

    /**
     * Get the current file ID
     * @returns {string|null} The current file ID
     */
    getFileId() {
        return this.fileId;
    }

    // ========================================
    // Position Storage
    // ========================================

    /**
     * Store the current playback position for the current file
     * @param {number} value - The position in seconds
     */
    storePosition(value) {
        if (!this.fileId) {
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
        if (!this.fileId) {
            return null;
        }
        const key = this._getPositionKey();
        const value = localStorage.getItem(key);
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
        if (!this.fileId) {
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
        if (!this.fileId) {
            return [];
        }
        const key = this._getBookmarksKey();
        const value = localStorage.getItem(key);
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
        if (!this.fileId) {
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
        if (!this.fileId) {
            return false;
        }
        const key = this._getBookmarksKey();
        return localStorage.getItem(key) !== null;
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

    // ========================================
    // Private Methods
    // ========================================

    /**
     * Get the localStorage key for position storage
     * @private
     * @returns {string} The position key
     */
    _getPositionKey() {
        return `fileId_${this.fileId}`;
    }

    /**
     * Get the localStorage key for bookmark storage
     * @private
     * @returns {string} The bookmarks key
     */
    _getBookmarksKey() {
        return `bookmarks_fileId_${this.fileId}`;
    }
}
