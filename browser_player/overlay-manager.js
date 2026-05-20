"use strict";

/**
 * Coordinates visibility and states for:
 * - scrim overlay
 * - menu and navigation panels
 * - menu and navigation toggle buttons
 */
class OverlayManager {

    /**
     * @param {Object} config - Configuration object
     * @param {HTMLElement} config.scrim - The scrim overlay element
     * @param {Menu} config.menu - Menu panel
     * @param {NavigationPanel} config.navigationPanel - Navigation panel
     * @param {HTMLElement} config.menuButton - The menu button element
     * @param {HTMLElement} config.navigationPanelButton - The navigation panel button element
     * @param {BookText} config.bookText - book text component
     * @param {RootAttributer} config.rootAttributer
     * @param {Function} config.onOverlayStart - Callback when overlay is shown () => void
     * @param {Function} config.onOverlayEnd - Callback when overlay is hidden () => void
     */
    constructor(config) {

        const requiredKeys = ["scrim", "menu", "menuButton", "navigationPanelButton", "navigationPanel", "bookText", "onOverlayStart", "onOverlayEnd", "rootAttributer"]
        const err = Util.validateObject(config, requiredKeys);
        if (err) {
            throw new Error(err)
        }

        this.scrim = config.scrim;
        this.menu = config.menu;
        this.navigationPanel = config.navigationPanel;
        this.menuButton = config.menuButton;
        this.navigationPanelButton = config.navigationPanelButton;
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
     * Toggle the navigation panel visibility
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
        this.navigationPanel.hide();
        this.collapseMenuButton();
        this.menu.update();        
        this.menu.show();
    }

    /**
     * Toggle the navigation panel visibility
     */
    toggleNavigationPanel() {
        if (!this.navigationPanel.isShowing) {
            this.showNavigationPanel();
        } else {
            this.hideAll();
        }
    }

    /**
     * Show the navigation panel
     */
    showNavigationPanel() {
        this.showScrim();
        this.menu.hide()
        this.bookText.removeBookmarkClasses(this.navigationPanel.indices); // efficient way of keeping bookText bookmarks up-to-date
        this.navigationPanel.updateAddButton(this.bookText.getCurrentIndex());
        this.navigationPanel.show()
    }

    /**
     * Hide scrim and all panels, restore body scroll
     */
    hideAll() {
        this.onOverlayEnd();
        ShowUtil.hide(this.scrim);
        this.menu.hide();
        this.navigationPanel.hide();
        this.bookText.addBookmarkClasses(this.navigationPanel.indices);
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
     * Show the navigation panel button
     */
    showNavigationPanelButton() {
        ShowUtil.show(this.navigationPanelButton);
    }

    /**
     * Hide the navigation panel button
     */
    hideNavigationPanelButton() {
        ShowUtil.hide(this.navigationPanelButton);
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
        this.navigationPanelButton.addEventListener('click', () => { this.toggleNavigationPanel(); });
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
