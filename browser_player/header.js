"use strict";

/**
 * Manages the header / top-of-page elements.
 * Namely: "load file" and "load url" inputs, plus.
 */
class Header {

    /**
     * @param {Object} config - Configuration object
     * @param {HTMLElement} config.mainEl - widget container element
     * @param {HTMLTemplateElement} config.helpTemplate - template for the help text
     * @param {Function} config.onUrlSubmit - Callback when URL is submitted (url) => void
     * @param {Function} config.onFileChange - Callback when file is selected (file) => void
     */
    constructor(config) {

        const requiredKeys = ["mainEl", "helpTemplate", "onUrlSubmit", "onFileChange"]
        const err = Util.validateObject(config, requiredKeys);
        if (err) {
            throw new Error(err)
        }

        this.mainEl = config.mainEl;

        this.fileInput = this.mainEl.querySelector("#loadFileInput");
        this.loadLocalButtonLabel = this.mainEl.querySelector("#loadLocalButtonLabel");
        this.urlInput = this.mainEl.querySelector("#loadUrlInput");
        this.loadUrlIcon = this.mainEl.querySelector("#loadUrlIcon");
        this.lastFileNameLabel = this.mainEl.querySelector("#lastFileNameLabel");
        this.lastFileNameText = this.mainEl.querySelector("#lastFileNameText");
        this.githubButton = this.mainEl.querySelector("#githubCorner");
        this.helpHolder = this.mainEl.querySelector("#helpHolder");
        
        const els = [this.fileInput, this.loadLocalButtonLabel, this.urlInput, this.loadUrlIcon, this.lastFileNameLabel, this.lastFileNameText, this.githubButton, this.helpHolder];
        for (const el of els) {
            if (!el) {
                throw new Error("Missing html element");
            }
        }
        
        this.helpTemplate = config.helpTemplate;
        this.onUrlSubmit = config.onUrlSubmit;
        this.onFileChange = config.onFileChange;

        this.urlInput.addEventListener('keydown', this._onLoadUrlKeyDown.bind(this));
        this.urlInput.addEventListener('blur', this._onLoadUrlBlur.bind(this));
        this.fileInput.addEventListener('change', this._onLoadFileChange.bind(this));
        this.fileInput.addEventListener('click', this._onLoadFileClick.bind(this));
        this.loadUrlIcon.addEventListener('click', this._onLoadUrlIconClick.bind(this));
    }

    doFileButtonClick() {
        this.fileInput.click();
        this.loadLocalButtonLabel.blur();
    }

    clearHelp() {
        this.helpHolder.innerHTML = "";
    }

    showHelpIfKeyboard() {
        this.clearHelp();
        if (Util.hasPersistentKeyboard()) {
            const help = this.helpTemplate.content.cloneNode(true);
            this.helpHolder.appendChild(help);
        }
    }
    
    // ========================================
    // Private Methods
    // ========================================

    /**
     * Handle keydown event on URL input
     * @param {KeyboardEvent} e
     * @private
     */
    _onLoadUrlKeyDown(e) {
        if (e.key === 'Enter') {
            const url = Util.undecodeUrl(this.urlInput.value.trim())
            if (url) {
                this.urlInput.blur();
                this.onUrlSubmit(url);
            }
        }
    }

    /**
     * Handle blur event on URL input
     * @private
     */
    _onLoadUrlBlur() {
        this.urlInput.value = '';
    }

    /**
     * Handle change event on file input
     * @private
     */
    _onLoadFileChange() {
        const file = this.fileInput.files[0];
        if (file) {
            this.onFileChange(file);
        }
    }

    /**
     * Handle click event on file input
     * @private
     */
    _onLoadFileClick() {
        const onFocusBack = () => {
            window.removeEventListener('focus', onFocusBack);
            setTimeout(() => {
                this.loadLocalButtonLabel.blur();
            }, 1);
        };
        window.addEventListener('focus', onFocusBack);
    }

    /**
     * Handle click event on URL info icon
     * @private
     */
    _onLoadUrlIconClick() {
        this.showLoadUrlInfo();
    }

    /**
     * Show information dialog about URL loading and CORS
     */
    showLoadUrlInfo() {
        let s = "For easy sharing...\n\n";
        s += "In the address bar, append ?url= followed by your hosted file's URL to share your audiobook with others.\n\n";
        s += "CORS info:\n\n";
        s += `When hosting audio files to be used by the player, remember to set "Access-Control-Allow-Origin" appropriately.\n\n`;
        s += `For example:\n`;
        s += `Access-Control-Allow-Origin: https://zeropointnine.github.io, http://192.168.1.2:8000\n`;
        s += `or: \n`;
        s += `Access-Control-Allow-Origin: *`;
        alert(s);
    }

    // ========================================
    // Last File Name Display
    // ========================================

    showLastFileName(fileName) {
        if (fileName) {
            this.lastFileNameText.textContent = fileName;
            this.lastFileNameLabel.style.display = "block";
        } else {
            this.lastFileNameText.textContent = "";
            this.lastFileNameLabel.style.display = "none";
        }
    }

    hideLastFileName() {
        this.lastFileNameLabel.style.display = "none";
        this.lastFileNameText.style.display = "none";
    }
}
