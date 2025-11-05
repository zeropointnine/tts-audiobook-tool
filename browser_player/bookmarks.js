/**
 * Acts as a 'controller' for the bookmark panel
 */
class BookmarkController {

    constructor(mainDiv) {
        this.mainDiv = mainDiv;
        this.bookmarkDesc = this.mainDiv.querySelector("#bookmarkDesc");
        this.addButton = this.mainDiv.querySelector("#bookmarkAddButton");
        this.listHolder = this.mainDiv.querySelector("#bookmarkList");
        this.indices = [];
        this.currentIndex = -1;

        this.addButton.addEventListener('click', () => { this._onAddButton() });
    }

    /**
     * Should be called when a new file is opened
     */
    init(textSegments, indices=[]) {

        this.textSegments = textSegments;
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
            this.addButton.classList.remove("disabled");
            this.bookmarkDesc.style.display = "block";
            const text = this.textSegments[this.currentIndex].text;
            this.bookmarkDesc.textContent = text
        } else {
            this.addButton.classList.add("disabled")
            this.bookmarkDesc.style.display = "none";
            this.bookmarkDesc.textContent = ""
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

    _updateList() {
        this.listHolder.replaceChildren();
        if (this.indices.length > 0) {
            for (const index of this.indices) {
                const itemEl = this._makeItemEl(index);
                this.listHolder.appendChild(itemEl);
            }
        } else {
            const noneItem = this._makeElement("<div style='margin-left:18px;'>No bookmarks added.</div>")
            this.listHolder.appendChild(noneItem);
        }
    }

    _makeItemEl(index) {

        const itemEl = this._makeElement(`<div class="bookmarkItem">`);

        const text = this.textSegments[index].text;
        const s = `<div class="bookmarkItemText" data-index="${index}" tabindex="-1">${text}</div>`;
        const textEl = this._makeElement(s);
        textEl.addEventListener("click", (e) => {
            const index = e.currentTarget.dataset.index;
            const event = new CustomEvent( "bookmarkSelect", { detail: { index: index } } )
            document.dispatchEvent(event)
        });
        const location = (index + 1) + "";
        const locationEl = this._makeElement(`<div class="bookmarkItemLocation">${location}</div>`)

        const deleteButtonEl = this._makeElement(`<div class="bookmarkItemDelete chromelessButton" data-index='${index}'><svg><use href="#iconDelete"></use></svg></div>`);
        deleteButtonEl.addEventListener('click', (e) => {
            const index = e.currentTarget.dataset.index;
            this.removeIndex(index);
        })

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
        document.dispatchEvent(new CustomEvent("bookmarkAdded"));
    }
}
