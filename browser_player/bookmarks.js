"use strict";

/**
 * Bookmarks panel widget
 * 
 * (Bookmark toggle button handled elsewhere)
 */
class Bookmarks {

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

        this.addButton = this.mainEl.querySelector("#bookmarkAddButton");
        this.addButtonDescLine = this.mainEl.querySelector("#bookmarkAddDescLine");
        this.addButtonDesc = this.mainEl.querySelector("#bookmarkAddDesc");
        this.addButtonLocation = this.mainEl.querySelector("#bookmarkAddLocation");
        this.listHolder = this.mainEl.querySelector("#bookmarkList");
        
        const els = [this.addButton, this.addButtonDescLine, this.addButtonDesc, this.addButtonLocation, this.listHolder];
        for (const el of els) {
            if (!el) {
                throw new Error("Missing html element");
            }
        }

        this.onSelected = config.onSelected;
        this.onChanged = config.onChanged;

        this.indices = [];
        this.currentIndex = -1;

        this.addButton.addEventListener('click', () => { this._onAddButton() });
        this.mainEl.addEventListener("click", (e) => { e.stopPropagation(); });
    }

    /**
     * Should be called when a new file is opened
     */
    init(textSegments, indices=[]) {
        this.textSegments = textSegments;
        this.initIndices(indices)
    }

    initIndices(indices) { 
        this.indices = [];
        if (indices) {
            for (const index of indices) {
                this.addIndex(index, false);
            }
        }
        this._updateList();
    }

    /**
     * Should be called when the bookmark panel is shown
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
        this.indices.sort();
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
                const itemEl = this._makeItemEl(index);
                this.listHolder.appendChild(itemEl);
            }
        } else {
            const noneItem = this._makeElement(`<div class="bookmarkItemNone">No bookmarks added.</div>`)
            this.listHolder.appendChild(noneItem);
        }
    }

    _makeItemEl(index) {

        // Item
        const itemEl = this._makeElement(`<div class="bookmarkItem">`);

        // Description text
        const text = this.textSegments[index].text;
        const s = `<div class="bookmarkItemText" data-index="${index}" tabindex="-1">${text}</div>`;
        const textEl = this._makeElement(s);
        textEl.addEventListener("click", this._onItemClick);

        // Location text
        const location = (index + 1) + "";
        const locationEl = this._makeElement(`<div class="bookmarkLocation">${location}</div>`)

        // Delete button
        const deleteButtonEl = this._makeElement(`<div class="bookmarkItemDelete chromelessButton" data-index='${index}'><svg><use href="#iconDelete"></use></svg></div>`);
        deleteButtonEl.addEventListener("click", (e) => { this._onItemDeleteButton(e) } );

        itemEl.appendChild(textEl);
        itemEl.appendChild(locationEl);
        itemEl.appendChild(deleteButtonEl);
        return itemEl;
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
