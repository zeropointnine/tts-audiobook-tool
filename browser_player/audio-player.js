class AudioPlayer {

    /**
     * Takes in a pre-existing <audio> element and an empty container.
     * The controls get programmatically added to the container.
     * Assumes css definitions as found in "audio-player.css"
     *
     * The class requires very little no "api", as it were.
     */
    constructor({ audioElement, container }) {

        this.audio = audioElement;
        this.container = container;
        this.audio.style.display = 'none'; // Hide original player

        this._createUi();
        this._bindEvents();

        // Initial state

        this._isPinned = false;

        this._updatePlayPauseButton();
        this._updateVolumeControls();
        this._updatePinButton();
    }

    isPinned() {
        return this._isPinned;
    }

    setPinned(b) {
        // Sets the icon type
        // (not the actual show/hide behavior, which this class does not address)
        this._isPinned = b;
        this._updatePinButton();
    }

    removePinButton() {
        this.pinBtn.remove();
    }

    makeButtonsInert(b) {
        if (b) {
            this.playPauseBtn.setAttribute("inert", "");
            this.volumeBtn.setAttribute("inert", "");
            this.pinBtn.setAttribute("inert", "");
        } else {
            this.playPauseBtn.removeAttribute("inert");
            this.volumeBtn.removeAttribute("inert");
            this.pinBtn.removeAttribute("inert");
        }
    }

    _createUi() {
        this.playerElement = document.createElement('div');
        this.playerElement.className = 'audio-player';

        this.playPauseBtn = document.createElement('button');
        this.playPauseBtn.className = 'play-pause-btn';

        this.progressBar = document.createElement('div');
        this.progressBar.className = 'progress-bar';
        this.progressBarThumb = document.createElement('div');
        this.progressBarThumb.className = 'progress-bar-thumb';
        this.progressBar.appendChild(this.progressBarThumb);

        this.timeDisplay = document.createElement('div');
        this.timeDisplay.className = 'time-display';
        this.timeDisplay.textContent = '0:00';

        this.timeTotal = document.createElement('div');
        this.timeTotal.className = 'time-total';
        this.timeTotal.textContent = '0:00';

        this.volumeControls = document.createElement('div');
        this.volumeControls.className = 'volume-controls';

        this.volumeBtn = document.createElement('button');
        this.volumeBtn.className = 'volume-btn';

        this.volumeSlider = document.createElement('div');
        this.volumeSlider.className = 'volume-slider';
        this.volumeSliderThumb = document.createElement('div');
        this.volumeSliderThumb.className = 'volume-slider-thumb';
        this.volumeSlider.appendChild(this.volumeSliderThumb);

        this.volumeControls.appendChild(this.volumeBtn);
        this.volumeControls.appendChild(this.volumeSlider);

        this.pinBtn = document.createElement('button');
        this.pinBtn.className = 'pin-btn';

        this.playerElement.appendChild(this.playPauseBtn);
        this.playerElement.appendChild(this.progressBar);
        this.playerElement.appendChild(this.timeDisplay);
        this.playerElement.appendChild(this.timeTotal);
        this.playerElement.appendChild(this.volumeControls);
        this.playerElement.appendChild(this.pinBtn);

        this.container.appendChild(this.playerElement);
    }

    _bindEvents() {
        // Audio element events
        this.audio.addEventListener('play', this._updatePlayPauseButton.bind(this));
        this.audio.addEventListener('pause', this._updatePlayPauseButton.bind(this));
        this.audio.addEventListener('timeupdate', this._updateTime.bind(this));
        this.audio.addEventListener('volumechange', this._updateVolumeControls.bind(this));
        this.audio.addEventListener('loadedmetadata', this._updateTime.bind(this));

        // Player UI events
        this.playPauseBtn.addEventListener('click', this._togglePlayPause.bind(this));
        this.volumeBtn.addEventListener('click', this._toggleMute.bind(this));

        // Seeking
        this.progressBar.addEventListener('click', this._onProgressBarClick.bind(this));
        this._makeDraggable(this.progressBarThumb, this.progressBar, (percent) => {
            this.audio.currentTime = this.audio.duration * percent;
        });

        // Volume
        this.volumeSlider.addEventListener('click', this._onVolumeSliderClick.bind(this));
        this._makeDraggable(this.volumeSliderThumb, this.volumeSlider, (percent) => {
            this.audio.volume = percent;
            if (this.audio.muted) {
                this.audio.muted = false;
            }
        });
    }

    _updatePlayPauseButton() {
        if (this.audio.paused) {
            this.playPauseBtn.classList.remove('playing');
            this.playPauseBtn.classList.add('paused');
        } else {
            this.playPauseBtn.classList.remove('paused');
            this.playPauseBtn.classList.add('playing');
        }
    }

    _updateTime() {
        let currentTime = this.audio.currentTime;
        let duration = this.audio.duration;
        if (!(duration > 0)) {
            currentTime = 0;
            duration = 0;
        }
        if (!(currentTime <= duration)) {
            currentTime = duration;
        }

        const x = this._timeToThumbX(currentTime, duration)
        this.progressBarThumb.style.left = `${x}px`;

        this.timeDisplay.textContent = `${this._formatTime(currentTime)}`;
        this.timeTotal.textContent = `\u00A0/ ${this._formatTime(duration)}`;
    }

    _updateVolumeControls() {

        // Icon
        if (this.audio.muted || this.audio.volume === 0) {
            this.volumeBtn.classList.add('muted');
            this.volumeBtn.classList.remove('unmuted');
        } else {
            this.volumeBtn.classList.remove('muted');
            this.volumeBtn.classList.add('unmuted');
        }

        // Thumb pos
        const barWidth = this.volumeSlider.offsetWidth;
        const thumbWidth = this.volumeSliderThumb.offsetWidth;
        if (barWidth == 0) {
            setTimeout(() => { this._updateVolumeControls(); }, 100); // yes well
            return;
        }
        if (this.audio.muted || this.audio.volume === 0) {
            this.volumeSliderThumb.style.left = '0px';
        } else {
            const thumbLeft = this.audio.volume * (barWidth - thumbWidth);
            this.volumeSliderThumb.style.left = `${thumbLeft}px`;
        }
    }

    _updatePinButton() {
        if (this._isPinned) {
            this.pinBtn.classList.remove('unpinned');
            this.pinBtn.classList.add('pinned');
        } else {
            this.pinBtn.classList.remove('pinned');
            this.pinBtn.classList.add('unpinned');
        }
    }

    _togglePlayPause() {
        if (this.audio.paused) {
            this.audio.play();
        } else {
            this.audio.pause();
        }
    }

    _toggleMute() {
        this.audio.muted = !this.audio.muted;
    }

    _onProgressBarClick(event) {
        const percent = this._clientXToPercent(event.clientX, this.progressBarThumb, this.progressBar);
        this.audio.currentTime = percent * this.audio.duration;
    }

    _onVolumeSliderClick(event) {
        const percent = this._clientXToPercent(event.clientX, this.volumeSliderThumb, this.volumeSlider);
        this.audio.volume = percent;
        if (this.audio.muted) {
            this.audio.muted = false;
        }
    }

    _makeDraggable(thumb, bar, callback) {
        let isDragging = false;

        const onMouseDown = (e) => {
            isDragging = true;
            thumb.style.transition = 'none';
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
            // Handle touch events
            document.addEventListener('touchmove', onMouseMove, { passive: false });
            document.addEventListener('touchend', onMouseUp);
            e.preventDefault();
        };

        const onMouseMove = (e) => {
            if (!isDragging) return;
            const event = e.touches ? e.touches[0] : e;

            // const rect = bar.getBoundingClientRect();
            // let percent = (event.clientX - rect.left) / rect.width;
            // percent = Math.max(0, Math.min(1, percent));

            const percent = this._clientXToPercent(event.clientX, thumb, bar);

            callback(percent);
            e.preventDefault();
        };

        const onMouseUp = () => {
            isDragging = false;
            thumb.style.transition = '';
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
            document.removeEventListener('touchmove', onMouseMove);
            document.removeEventListener('touchend', onMouseUp);
        };

        thumb.addEventListener('mousedown', onMouseDown);
        thumb.addEventListener('touchstart', onMouseDown, { passive: false });
    }

    // ---

    _formatTime(seconds) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        const parts = [];
        if (h > 0) {
            parts.push(h);
            parts.push(m.toString().padStart(2, '0'));
        } else {
            parts.push(m);
        }
        parts.push(s.toString().padStart(2, '0'));
        return parts.join(':');
    }

    _timeToThumbX(currentTime, duration) {
        const percent = duration ? currentTime / duration : 0;
        const thumbWidth = this.progressBarThumb.offsetWidth;
        const width = this.progressBar.offsetWidth - thumbWidth;
        const thumbLeft = percent * width;
        this.progressBarThumb.style.left = `${thumbLeft}px`;
    }

    _clientXToPercent(clientX, thumb, bar) {
        const thumbWidth = thumb.offsetWidth;
        const rect = bar.getBoundingClientRect();
        const left = rect.left + thumbWidth / 2;
        const width = rect.width - thumbWidth;
        let percent = width ? (clientX - left) / width : 0;
        percent = Math.max(0, Math.min(1, percent));
        return percent;
    }
}
