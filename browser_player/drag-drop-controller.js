"use strict";

/**
 * Manages drag-and-drop functionality for loading audio files.
 * Handles the drag overlay visibility and file drop events.
 */
class DragDropController {

    // Supported file extensions
    static VALID_EXTENSIONS = ['.m4a', '.m4b', '.flac'];

    /**
     * @param {Object} config - Configuration object
     * @param {HTMLElement} config.overlay - The drag overlay element
     * @param {Function} config.onDropFile - Callback when a file is dropped (file) => void
     * @param {Function} config.isLoading - Function to check if app is loading () => boolean
     */
    constructor(config) {
        const err = Util.validateObject(config, ["overlay", "onDropFile", "isLoading"]);
        if (err) {
            throw new Error(err)
        }

        this.overlay = config.overlay;
        this.onDropFile = config.onDropFile;
        this.isLoading = config.isLoading;

        // Internal state
        this.dragCounter = 0;
        this.isIgnoreDrag = false;

        // Initialize listeners
        this._initListeners();
    }

    // ========================================
    // Private Methods
    // ========================================

    /**
     * Initialize drag-and-drop event listeners
     * @private
     */
    _initListeners() {
        window.addEventListener('dragenter', this._onDragEnter.bind(this));
        window.addEventListener('dragleave', this._onDragLeave.bind(this));
        window.addEventListener('dragover', this._onDragOver.bind(this));
        window.addEventListener('drop', this._onDrop.bind(this));
    }

    /**
     * Handle dragenter event
     * @param {DragEvent} e
     * @private
     */
    _onDragEnter(e) {
        e.preventDefault();
        this.isIgnoreDrag = this.isLoading();
        this.dragCounter++;
        if (this.isIgnoreDrag) {
            return;
        }
        this.overlay.style.display = 'block';
    }

    /**
     * Handle dragleave event
     * @param {DragEvent} e
     * @private
     */
    _onDragLeave(e) {
        e.preventDefault();
        this.dragCounter--;
        if (this.dragCounter === 0) {
            this.overlay.style.display = 'none';
            if (this.isIgnoreDrag) {
                this.isIgnoreDrag = false;
            }
        }
    }

    /**
     * Handle dragover event
     * @param {DragEvent} e
     * @private
     */
    _onDragOver(e) {
        e.preventDefault();
    }

    /**
     * Validate file extension
     * @param {File} file - The file to validate
     * @returns {boolean} True if file has a valid extension
     * @private
     */
    _isValidFile(file) {
        if (!file || !file.name) {
            return false;
        }
        const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));
        return DragDropController.VALID_EXTENSIONS.includes(ext);
    }

    /**
     * Handle drop event
     * @param {DragEvent} e
     * @private
     */
    _onDrop(e) {
        e.preventDefault();
        this.dragCounter = 0;
        this.overlay.style.display = 'none';

        if (this.isIgnoreDrag) {
            this.isIgnoreDrag = false;
            return;
        }

        const file = e.dataTransfer.files[0];
        if (!this._isValidFile(file)) {
            alert("Unsupported file type. Please use .m4a, .m4b, or .flac files.");
            return;
        }
        this.onDropFile(file);
    }
}
