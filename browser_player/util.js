/**
 * Utility functions for the audiobook player
 */
class Util {
    /**
     * Escapes HTML characters to prevent XSS
     */
    static escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;")    
    }

    /**
     * Generates a 32-bit hash of an object using FNV-1a algorithm
     */
    static getObjectHash(obj) {
    
        // Stable string representation of object
        const str = JSON.stringify(obj, Object.keys(obj).sort());

        // FNV-1a Hashing Algorithm
        let hash = 2166136261; // FNV offset basis
        for (let i = 0; i < str.length; i++) {
            hash ^= str.charCodeAt(i);
            // Perform 32-bit integer multiplication
            hash = Math.imul(hash, 16777619); // FNV prime
        }
        // Convert to unsigned hex string
        return (hash >>> 0).toString(16);
    }

    /**
     * Returns true if device has a touch interface, hand-wavey test
     */
    static isTouchDevice() {
        return !window.matchMedia("(pointer:fine)").matches
    }

    /**
     * Returns true if device has persistent keyboard support, hand-wavey test
     */
    static hasPersistentKeyboard() {
        const hasFinePointer = window.matchMedia("(pointer: fine)").matches;
        const hasHoverSupport = window.matchMedia("(hover: hover)").matches;
        return (hasFinePointer && hasHoverSupport);
    }

    /**
     * Splits a string into leading whitespace, content, and trailing whitespace
     */
    static splitWhitespace(str) {
        // Match leading whitespace, content, and trailing whitespace
        str = str || "";
        const match = str.match(/^(\s*)(.*?)(\s*)$/);
        if (!match) {
            // This should not happen with the current regex,
            // but as a safeguard, return empty parts if match fails.
            return {
                before: "",
                content: str,
                after: ""
            };
        }
        return {
            before: match[1],
            content: match[2],
            after: match[3]
        };
    }

    /**
     * Converts milliseconds to a human-readable string (e.g., "1m30s" or "30s")
     */
    static msToString(milliseconds) {
        const totalSeconds = Math.floor(milliseconds / 1000);
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = totalSeconds % 60;
        if (minutes > 0) {
            return `${minutes}m${seconds}s`;
        } else {
            return `${seconds}s`;
        }
    }

    /**
     * Returns a url-decoded url string if it is url-encoded, else identity.
     * 
     * Use case is user copy-pasting the url queryparam value in the address bar, 
     * which is in url-encoded format
     */
    static undecodeUrl(url) {
        try {
            const decoded = decodeURIComponent(url);
            // If the decoded version looks more like a URL (contains ://), use it
            if (decoded.includes('://') && !url.includes('://')) {
                url = decoded;
            }
        } catch (e) {
            // Use original value
        }
        return url;
    }

    /**
     * Verifies that only the given `requiredKeys` exist in `object`, and with non-null values.
     * With additional `optionalKeys`.`
     * Returns error message string or empty string
     */
    static validateObject(object, requiredKeys, optionalKeys=[]) {
        for (const key of requiredKeys) {
            if (object[key] === null || object[key] === undefined) {
                return `missing key in object: ${key}`;
            }
        }
        for (const key in object) {
            if (!requiredKeys.includes(key) && !optionalKeys.includes(key)) {
                return `unexpected key in object: ${key}`;
            }
        }
        return "";
    }
}
