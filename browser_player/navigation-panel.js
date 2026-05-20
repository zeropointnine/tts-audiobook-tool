"use strict";

/**
 * Navigation panel widget
 * 
 * (Navigation panel toggle button handled elsewhere)
 */
class NavigationPanel {

    activeTab = "sections";

    /**
     * @param {Object} config - Configuration object
     * @param {Object} config.mainEl - 
     * @param {Object} config.onSelected - 
     * @param {Object} config.onChanged - 
     */
    constructor(config) {

        const err = Util.validateObject(config, ["mainEl", "onSelected", "onChanged"]);
        if (err) {
            throw new Error(err)
        }

        this.mainEl = config.mainEl;

        this.addButton = this.mainEl.querySelector("#bookmarksAddButton");
        this.addButtonDescLine = this.mainEl.querySelector("#bookmarksAddDescLine");
        this.addButtonDesc = this.mainEl.querySelector("#bookmarksAddDesc");
        this.addButtonLocation = this.mainEl.querySelector("#bookmarksAddLocation");
        this.sectionsTab = this.mainEl.querySelector("#sectionsTab");
        this.bookmarksTab = this.mainEl.querySelector("#bookmarksTab");
        this.sectionsView = this.mainEl.querySelector("#sectionsView");
        this.bookmarksView = this.mainEl.querySelector("#bookmarksView");
        this.sectionList = this.mainEl.querySelector("#sectionList");
        this.listHolder = this.mainEl.querySelector("#bookmarksList");
        
        const els = [
            this.addButton,
            this.addButtonDescLine,
            this.addButtonDesc,
            this.addButtonLocation,
            this.sectionsTab,
            this.bookmarksTab,
            this.sectionsView,
            this.bookmarksView,
            this.sectionList,
            this.listHolder,
        ];
        for (const el of els) {
            if (!el) {
                throw new Error("Missing html element");
            }
        }

        this.onSelected = config.onSelected;
        this.onChanged = config.onChanged;

        this.indices = [];
        this.sections = [];
        this.currentIndex = -1;

        this.addButton.addEventListener('click', () => { this._onAddButton() });
        this.sectionsTab.addEventListener('click', () => { this.selectTab("sections"); });
        this.bookmarksTab.addEventListener('click', () => { this.selectTab("bookmarks"); });
        this.mainEl.addEventListener("click", (e) => { e.stopPropagation(); });
    }

    /**
     * Should be called when a new file is opened
     */
    init(textSegments, indices = [], sections = []) {
        this.textSegments = textSegments;
        this.sections = Array.isArray(sections) ? sections : [];
        this._updateTabVisibility();
        this.selectTab(this._shouldShowTabs() ? "sections" : "bookmarks");
        this.initIndices(indices)
        this._updateSectionList();
    }

    initIndices(indices) { 
        this.indices = [];
        if (indices) {
            for (const index of indices) {
                this.addIndex(parseInt(index), false);
            }
        }
        this._updateList();
    }

    /**
     * Should be called when the navigation panel is shown
     */
    updateAddButton(index) {

        this.currentIndex = (index >= 0 && index < this.textSegments.length) ? index : -1;

        if (this.currentIndex > -1) {
            this.addButton.style.display = "block";
            const desc_text = this.textSegments[this.currentIndex].text;
            this.addButtonDesc.textContent = desc_text
            const loc_text = `${index + 1}/${this.textSegments.length}`
            this.addButtonLocation.textContent = loc_text;
        } else {
            this.addButton.style.display = "none";
        }
    }

    addIndex(index, andUpdate=true) {
        if (!this.isValidIndex(index)) {
            return;
        }
        const exists = (this.indices.indexOf(index) > -1);
        if (exists) {
            return;
        }
        this.indices.push(index);
        this.indices.sort((a, b) => a - b);
        if (andUpdate) {
            this._updateList();
        }
    }

    isValidIndex(index) {
        if (index < 0 || index >= this.textSegments.length) {
            return false;
        }
        const duration =  this.textSegments[index]["time_end"] - this.textSegments[index]["time_start"];
        if (!(duration > 0)) {
            return false
        }
        return true;
    }

    removeIndex(indexValue) {
        const a = [];
        for (const i of this.indices) {
            if (i != indexValue) {
                a.push(i);
            }
        }
        this.indices = a;
        this._updateList();
    }

    show() {
       this.selectTab(this.activeTab || "sections");
       ShowUtil.show(this.mainEl);
    }

    hide() {
       ShowUtil.hide(this.mainEl);
    }

    get isShowing() {
        return ShowUtil.isShowing(this.mainEl);
    }

    _updateList() {
        this.listHolder.replaceChildren();
        if (this.indices.length > 0) {
            for (const index of this.indices) {
                const itemEl = this._makeBookmarkItemEl(index);
                this.listHolder.appendChild(itemEl);
            }
        } else {
            const noneItem = this._makeElement(`<div class="navigationItemNone">No bookmarks added.</div>`)
            this.listHolder.appendChild(noneItem);
        }
    }

    _updateSectionList() {
        this.sectionList.replaceChildren();

        if (this.sections.length === 0) {
            const noneItem = this._makeElement(`<div class="navigationItemNone">No sections available.</div>`);
            this.sectionList.appendChild(noneItem);
            return;
        }

        let numRendered = 0;
        for (const [i, section] of this.sections.entries()) {
            const itemEl = this._makeSectionItemEl(section, i);
            if (itemEl) {
                this.sectionList.appendChild(itemEl);
                numRendered += 1;
            }
        }

        if (numRendered === 0) {
            const noneItem = this._makeElement(`<div class="navigationItemNone">No sections available.</div>`);
            this.sectionList.appendChild(noneItem);
        }
    }

    selectTab(tabName) {
        if (!this._shouldShowTabs()) {
            this.activeTab = "bookmarks";
            this.sectionsView.style.display = "none";
            this.bookmarksView.style.display = "flex";
            this.mainEl.style.paddingTop = "18px";
            return;
        }

        this.activeTab = (tabName === "bookmarks") ? "bookmarks" : "sections";

        const isSections = this.activeTab === "sections";
        this.sectionsTab.classList.toggle("selected", isSections);
        this.bookmarksTab.classList.toggle("selected", !isSections);
        this.sectionsTab.setAttribute("aria-selected", isSections ? "true" : "false");
        this.bookmarksTab.setAttribute("aria-selected", isSections ? "false" : "true");
        this.sectionsView.style.display = isSections ? "flex" : "none";
        this.bookmarksView.style.display = isSections ? "none" : "flex";
        this.mainEl.style.paddingTop = "18px";

        setTimeout(() => {
            if (document.activeElement === this.sectionsTab || document.activeElement === this.bookmarksTab) {
                document.activeElement.blur();
            }
        }, 1);
    }

    _shouldShowTabs() {
        return this.sections.length > 1;
    }

    _updateTabVisibility() {
        const shouldShowTabs = this._shouldShowTabs();
        this.sectionsTab.parentElement.style.display = shouldShowTabs ? "flex" : "none";
        this.mainEl.style.paddingTop = shouldShowTabs ? "18px" : "0px";
    }

    _makeBookmarkItemEl(index) {

        // Item
        const itemEl = this._makeElement(`<div class="navigationItem">`);

        // Description text
        const text = this.textSegments[index].text;
        const s = `<div class="navigationItemText" data-index="${index}" tabindex="-1">${text}</div>`;
        const textEl = this._makeElement(s);
        textEl.addEventListener("click", this._onItemClick);

        // Location text
        const location = (index + 1) + "";
        const locationEl = this._makeElement(`<div class="navigationItemLocation">${location}</div>`)

        // Delete button
        const deleteButtonEl = this._makeElement(`<div class="bookmarksItemDelete chromelessButton" data-index='${index}'><svg><use href="#iconDelete"></use></svg></div>`);
        deleteButtonEl.addEventListener("click", (e) => { this._onItemDeleteButton(e) } );

        itemEl.appendChild(textEl);
        itemEl.appendChild(locationEl);
        itemEl.appendChild(deleteButtonEl);
        return itemEl;
    }

    _makeSectionItemEl(section, sectionIndex) {
        const targetIndex = this._getSectionPlayableIndex(section);
        if (targetIndex < 0) {
            return null;
        }

        const title = section.title || `Section ${sectionIndex + 1}`;
        const itemEl = this._makeElement(`<div class="navigationItem">`);
        const textEl = this._makeElement(`<div class="navigationItemText" data-index="${targetIndex}" tabindex="-1">${Util.escapeHtml(title)}</div>`);
        textEl.addEventListener("click", this._onItemClick);

        const location = `${targetIndex + 1}`;
        const locationEl = this._makeElement(`<div class="navigationItemLocation">${location}</div>`);

        itemEl.appendChild(textEl);
        itemEl.appendChild(locationEl);
        return itemEl;
    }

    _getSectionPlayableIndex(section) {
        if (!section || !Number.isInteger(section.start_index) || !Number.isInteger(section.end_index)) {
            return -1;
        }

        const startIndex = Math.max(0, section.start_index);
        const endIndex = Math.min(section.end_index, this.textSegments.length);

        for (let i = startIndex; i < endIndex; i++) {
            if (this.isValidIndex(i)) {
                return i;
            }
        }

        return this.isValidIndex(startIndex) ? startIndex : -1;
    }

    _makeElement(htmlString) {
        const temp = document.createElement("div");
        temp.innerHTML = htmlString;
        return temp.firstElementChild;
    }

    _onAddButton() {
        const exists = (this.indices.indexOf(this.currentIndex) > -1);
        if (exists) {
            return;
        }
        this.addIndex(this.currentIndex);
        this.onChanged();
    }

    _onItemClick = (e) => {
        let index = e.currentTarget.dataset.index;
        index = parseInt(index);
        this.onSelected(index);
    }

    _onItemDeleteButton(e) {
        const index = e.currentTarget.dataset.index;
        this.removeIndex(index);
        this.onChanged();
    }
}
