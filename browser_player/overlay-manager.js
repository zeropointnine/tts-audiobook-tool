"use strict";

/**
 * Coordinates visibility and states for:
 * - scrim overlay
 * - menu and bookmark panels
 * - menu and bookmark toggle buttons
 */
class OverlayManager {

    /**
     * @param {Object} config - Configuration object
     * @param {HTMLElement} config.scrim - The scrim overlay element
     * @param {Menu} config.menu - Menu panel
     * @param {Bookmarks} config.bookmarks - Bookmark panel
     * @param {HTMLElement} config.menuButton - The menu button element
     * @param {HTMLElement} config.bookmarkButton - The bookmark button element
     * @param {BookText} config.bookText - book text component
     * @param {RootAttributer} config.rootAttributer
     * @param {Function} config.onOverlayStart - Callback when overlay is shown () => void
     * @param {Function} config.onOverlayEnd - Callback when overlay is hidden () => void
     */
    constructor(config) {

        const requiredKeys = ["scrim", "menu", "menuButton", "bookmarkButton", "bookmarks", "bookText", "onOverlayStart", "onOverlayEnd", "rootAttributer"]
        const err = Util.validateObject(config, requiredKeys);
        if (err) {
            throw new Error(err)
        }

        this.scrim = config.scrim;
        this.menu = config.menu;
        this.bookmarks = config.bookmarks;
        this.menuButton = config.menuButton;
        this.bookmarkButton = config.bookmarkButton;
        this.bookText = config.bookText;
        this.onOverlayStart = config.onOverlayStart;
        this.onOverlayEnd = config.onOverlayEnd;
        this.rootAttributer = config.rootAttributer;

        // Initialize listeners
        this._initListeners();
    }

    /**
     * Check if the scrim is currently showing
     * @returns {boolean}
     */
    isScrimShowing() {
        return ShowUtil.isShowing(this.scrim);
    }

    /**
     * Show the scrim overlay and lock body scroll
     */
    showScrim() {
        this.onOverlayStart();
        ShowUtil.show(this.scrim);
    }

    /**
     * Toggle the bookmark panel visibility
     */
    toggleMenu() {
        if (!this.menu.isShowing) {
            this.showMenu();
        } else {
            this.hideAll();
        }
    }

    /**
     * Show the menu panel
     */
    showMenu() {
        this.showScrim();
        this.bookmarks.hide();
        this.collapseMenuButton();
        this.menu.update();        
        this.menu.show();
    }

    /**
     * Toggle the bookmark panel visibility
     */
    toggleBookmarks() {
        if (!this.bookmarks.isShowing) {
            this.showBookmarks();
        } else {
            this.hideAll();
        }
    }

    /**
     * Show the bookmark panel
     */
    showBookmarks() {
        this.showScrim();
        this.menu.hide()
        this.bookText.removeBookmarkClasses(this.bookmarks.indices); // efficient way of keeping bookText bookmarks up-to-date
        this.bookmarks.updateAddButton(this.bookText.getCurrentIndex());
        this.bookmarks.show()
    }

    /**
     * Hide scrim and all panels, restore body scroll
     */
    hideAll() {
        this.onOverlayEnd();
        ShowUtil.hide(this.scrim);
        this.menu.hide();
        this.bookmarks.hide();
        this.bookText.addBookmarkClasses(this.bookmarks.indices);
    }

    // ========================================
    // Button Visibility Methods
    // ========================================

    /**
     * Collapse the menu button by setting the collapse attribute on root
     */
    collapseMenuButton() {
        this.rootAttributer.set("data-ui-panel-button-collapse", "true");
    }

    /**
     * Show the bookmark button
     */
    showBookmarkButton() {
        ShowUtil.show(this.bookmarkButton);
    }

    /**
     * Hide the bookmark button
     */
    hideBookmarkButton() {
        ShowUtil.hide(this.bookmarkButton);
    }

    // ========================================
    // Private Methods
    // ========================================

    /**
     * Initialize event listeners
     * @private
     */
    _initListeners() {
        this.scrim.addEventListener("click", this._onScrimClick.bind(this));
        this.menuButton.addEventListener('click', () => { this.toggleMenu() });
        this.bookmarkButton.addEventListener('click', () => { this.toggleBookmarks(); });
    }

    /**
     * Handle scrim click event
     * @param {Event} e
     * @private
     */
    _onScrimClick(e) {
        e.stopPropagation();
        this.hideAll();
    }
}
