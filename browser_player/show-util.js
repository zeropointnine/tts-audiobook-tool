/**
 * Manages the visibility (ie, in-out transitions) of elements that utilize the "showing" css definition
 */
class ShowUtil {

    /**
     * Show an element with transition
     * @param {HTMLElement} element - The element to show
     * @param {string} displayValue - The display value (default: "block")
     * @private
     */
    static show(element, displayValue = "block") {
        element.removeEventListener('transitionend', this._onHideDisplayNone);
        element.style.display = displayValue;
        element.offsetHeight; // Force reflow
        element.classList.add("showing");
    }

    /**
     * Hide an element with transition
     * @param {HTMLElement} element - The element to hide
     * @param {boolean} shouldDisplayNone - Whether to set display: none after transition (default: true)
     * @private
     */
    static hide(element, shouldDisplayNone = true) {
        element.removeEventListener('transitionend', this._onHideDisplayNone);
        if (shouldDisplayNone) {
            element.addEventListener('transitionend', this._onHideDisplayNone, { once: true });
        }
        element.classList.remove("showing");
    }

    /**
     * Handle transition end to set display: none
     * @param {Event} e
     * @private
     */
    static _onHideDisplayNone(e) {
        e.currentTarget.style.display = "none";
    }

    /**
     * Check if an element is currently "showing" / visible
     * @param {HTMLElement} element
     * @returns {boolean}
     * @private
     */
    static isShowing(element) {
        return element.classList.contains("showing");
    }


}
