/**
 * Utility to get/set attributes on the root DOM element
 * for the purpose of affecting css selectors.
 * Plus "cycle" utility function.
 */
class RootAttributer {

    constructor(root, storageController) {
        if (!root || !storageController) {
            throw new Error("Missing param")
        }
        this.root = root
        this.storageController = storageController
    }

    set(attrib, value) {
        if (value === null) {
            this.root.removeAttribute(attrib)
        } else {
            this.root.setAttribute(attrib, value);
        }
    }

    /**
     * Get the value of a root attribute
     * @param {string} attrib - The attribute name
     * @returns {string} The attribute value or empty string if not set
     */
    get(attrib) {
        return this.root.getAttribute(attrib) || "";
    }

    /**
     * Cycle through possible values for a root data attribute
     * @param {string} attribute - The attribute name (e.g., "data-theme")
     * @param {Array<string>} values - Array of possible values to cycle through
     * @param {boolean} andStore - Whether to persist the changed setting to storage
     * @returns {string} The new attribute value (empty string if cleared)
     */
    cycle(attribute, values, andStore = true) {
    
        if (!Array.isArray(values) || values.length == 0) {
            throw new Error(`Bad value: ${values}`);
        }

        const currentValue = this.get(attribute);
        let currentIndex = values.indexOf(currentValue);

        let nextIndex;
        if (currentIndex == -1) {
            nextIndex = 0;
        } else {
            nextIndex = currentIndex + 1;
            if (nextIndex >= values.length) {
                nextIndex = -1;
            }
        }

        let targetValue;
        if (nextIndex == -1) {
            targetValue = "";
        } else {
            targetValue = values[nextIndex];
        }

        if (targetValue == "") {
            this.set(attribute, null);
        } else {
            this.set(attribute, targetValue);
        }

        if (andStore) {
            this.storageController.saveSetting(attribute, targetValue);
        }

        return targetValue;
    }

}