(function() {

    const DEFAULT_FADE_DELAY = 1000;

    const root = document.documentElement;

    let loadFileInput = null;
    let fileNameDiv = null;
    let playerHolder = null;
    let player = null;
    let textHolder = null;
    let loadLocalButtonLabel = null;
    let loadUrlInput = null;

    let rawText = "";
    let timedTextSegments = [];
    let selectedSpan = null;
    let intervalId = -1;
    let fadeOutId = -1;
    let isStarted = false;
    let isPlayerHover = false;
    let isPlayerFocused = false;
    let fadeOutValue = 0.0;

    function init() {

        loadFileInput = document.getElementById('loadFileInput');
        fileNameDiv = document.getElementById('fileName')
        playerHolder = document.getElementById('playerHolder');
        player = document.getElementById('player');
        textHolder = document.getElementById('textHolder');
        loadLocalButtonLabel = document.getElementById("loadLocalButtonLabel");
        loadUrlInput = document.getElementById('loadUrlInput');

        if (!matchMedia('(pointer:fine)').matches) {
            // Treat as touch device
            // Disable player fadeout
            fadeOutValue = 1.0;
        }

        loadUrlInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                const url = loadUrlInput.value.trim()
                if (url) {
                    loadUrlInput.blur();
                    loadFlacOrMp4(url);
                }
            }
        });

        loadUrlInput.addEventListener('blur', () => {
            loadUrlInput.value = '';
        });

        loadFileInput.addEventListener('change', async () => {
            const file = loadFileInput.files[0];
            if (file) {
                loadFlacOrMp4(file);
            }
        });

        player.addEventListener('play', function() {
            root.setAttribute("data-player-status", "play");
            // ensures scroll to current segment
            selectedSpan = null;
        });

        player.addEventListener('pause', function() {
            root.setAttribute("data-player-status", "pause");
        });

        playerHolder.addEventListener('mouseenter', () => {
            isPlayerHover = true;
            onPlayerIxChange();
        });

        playerHolder.addEventListener('mouseleave', () => {
            isPlayerHover = false;
            onPlayerIxChange();
        });

        // Focus and blur detection (bubbles)
        playerHolder.addEventListener('focusin', () => {
            isPlayerFocused = true;
            onPlayerIxChange();
        });
        playerHolder.addEventListener('focusout', () => {
            setTimeout(() => { // nb, required
            if (!playerHolder.contains(document.activeElement)) {
                isPlayerFocused = false;
                onPlayerIxChange();
            }
            }, 0);
        });

        textHolder.addEventListener('click', onTextClick);

        document.addEventListener("keydown", onKeyDown);

        // When the queryparam is "url", run the function "loadFlacOrMp4" using the value.
        const urlParams = new URLSearchParams(window.location.search);
        const url = urlParams.get('url');
        if (url) {
            loadFlacOrMp4(url);
        }

        initButtons();
    }

    function initButtons() {

        const uiToggleButton = document.getElementById('uiToggleButton');
        uiToggleButton.addEventListener('click', (e) => {
            console.log('button')
            if (uiOverlay.style.display !== 'block') {
                uiOverlay.style.display = 'block';
            } else {
                uiOverlay.style.display = 'none';
            }
        });

        const uiOverlay = document.getElementById("uiOverlay");
        uiOverlay.addEventListener("click", (e) => {
            e.stopPropagation();
            uiOverlay.style.display = "none";
        });

        const uiPanel = document.getElementById("uiPanel")
        uiPanel.addEventListener("click", (e) => {
            e.stopPropagation()
            console.log('panel')
        });

        // Scroll to top
        document.getElementById('scrollTopButton').addEventListener('click', (e) => {
            e.stopPropagation();
            window.scrollTo(0, 0);
        });

        // Color theme
        themeButton = document.getElementById('themeButton');
        themeButton.addEventListener('click', (e) => {
            e.stopPropagation();
            value = cycleRootAttribute("data-theme", ["dark"])
            setChildVisibleWhen(themeButton, "value", value);
        });
        value = setRootAttributeFromLocalStorage("data-theme");
        setChildVisibleWhen(themeButton, "value", value);


        // Font size
        document.getElementById('fontSizeButton')
        fontSizeButton.addEventListener('click', (e) => {
            e.stopPropagation();
            value = cycleRootAttribute("data-text-size", ["medium", "small"]);
            setChildVisibleWhen(fontSizeButton, "value", value);
        });
        value = setRootAttributeFromLocalStorage("data-text-size");
        setChildVisibleWhen(fontSizeButton, "value", value);

        // Segment colors
        document.getElementById('segmentColorsButton').addEventListener('click', (e) => {
            e.stopPropagation();
            cycleRootAttribute("data-segment-colors", ["blue"])
        });
        setRootAttributeFromLocalStorage("data-segment-colors");
    }

    async function loadFlacOrMp4(fileOrUrl) {
        clear();

        result = await loadAppMetadata(fileOrUrl);
        if (!result) {
            alert("No tts-audiobook-tool metadata found");
            return;
        }

        // TEMP
        // result = {
        //     raw_text: "hello",
        //     text_segments: [
        //         {
        //             index_start: 0,
        //             index_end: 4,
        //             time_start: 2.0,
        //             time_end: 3.0
        //         }
        //     ]
        //  }

        start(fileOrUrl, result["raw_text"], result["text_segments"]);
    }

    function clear() {
        isStarted = false;
        playerHolder.style.display = "none";
        fileNameDiv.style.display = "none"
        textHolder.style.display = "none";
        clearInterval(intervalId)
        player.src = null;
        selectedSpan = null;
        root.setAttribute("data-player-status", "none");
    }

    function start(fileOrUrl, pRawText, pTimedTextSegments) {

        let file = null;
        let url = null;
        if (typeof fileOrUrl === "string") {
            url = fileOrUrl;
        } else {
            file = fileOrUrl;
        }

        rawText = pRawText
        timedTextSegments = pTimedTextSegments

        fileNameDiv.style.display = "block"
        fileNameDiv.textContent = file ? file.name : url

        populateText()

        playerHolder.style.display = "block";
        showPlayerAndFade();

        player.src = file ? URL.createObjectURL(file) : url
        playerPlay();

        if (document.activeElement && document.activeElement.blur) {
            document.activeElement.blur();
        }

        intervalId = setInterval(loop, 50);
        isStarted = true;
    }

    function onTextClick(event) {

        isSegment = (event.target.tagName === 'SPAN' && event.target.id.startsWith('segment-'));
        if (!isSegment) {
            return;
        }

        const clickedSpan = event.target;
        const segmentIndex = parseInt(clickedSpan.id.split('-')[1]);

        if (clickedSpan == selectedSpan) {
            // Toggle play
            if (player.paused) {
                playerPlay();
            } else {
                player.pause()
            }
        } else {
            seekBySegmentIndex(segmentIndex);
            showPlayerAndFade();
        }
    }

    // --------------------------------------
    function onKeyDown(event) {

        if (event.target.tagName == "INPUT") {
            return;
        }
        // console.log(event.key);

        if (event.key === "Enter" || event.key === " ") {
            if (document.activeElement == loadLocalButtonLabel) {
                loadFileInput.click();
                event.preventDefault();
                loadLocalButtonLabel.blur();
            }
            return;
        }

        if (!isStarted) {
            return;
        }
        switch (event.key) {
            case "Escape":
                if (player.paused) {
                    playerPlay();
                } else {
                    player.pause();
                }
                event.preventDefault();
                break;
            case "[":
                seekPreviousSegment();
                break;
            case "]":
                seekNextSegment();
                break;
        }
    }

    function populateText() {

        let contentHtml = '';
        let lastIndex = 0;

        timedTextSegments.forEach((segment, i) => {
            // Add text before the current segment (if any)
            if (segment.index_start > lastIndex) {
                contentHtml += escapeHtml(rawText.substring(lastIndex, segment.index_start));
            }
            // Add the current segment wrapped in a span
            const segmentText = rawText.substring(segment.index_start, segment.index_end);
            contentHtml += `<span id="segment-${i}">${escapeHtml(segmentText)}</span>`;
            lastIndex = segment.index_end;
        });

        // Add any remaining text after the last segment
        if (lastIndex < rawText.length) {
            contentHtml += escapeHtml(rawText.substring(lastIndex));
        }

        textHolder.innerHTML = contentHtml;
        textHolder.style.display = "block";
    }

    function loop() {

        let span = getCurrentSpan()

        // Update highlighting only if the active segment has changed
        if (selectedSpan !== span) {
            if (selectedSpan) {
                selectedSpan.classList.remove('highlight');
            }
            if (span) {
                span.classList.add('highlight');
                // Scroll the new highlighted span into view, centered
                span.scrollIntoView({
                    behavior: 'smooth',
                    block: 'center', // 'start', 'center', 'end', or 'nearest'
                    inline: 'nearest' // 'start', 'center', 'end', or 'nearest'
                });
            }
            selectedSpan = span;
        }
    }

    function seekPreviousSegment() {
        i = getCurrentSegmentIndex()
        if (i == -1) {
            return;
        }
        i = i - 1;
        if (i < 0) {
            return;
        }
        seekBySegmentIndex(i)
    }

    function seekNextSegment() {
        i = getCurrentSegmentIndex()
        if (i == -1) {
            return;
        }
        i = i + 1;
        if (i >= timedTextSegments.length) {
            return;
        }
        seekBySegmentIndex(i)
    }

    function seekBySegmentIndex(i) {
        if (selectedSpan) {
            selectedSpan.classList.remove('highlight');
            selectedSpan = null;
        }

        targetTime = timedTextSegments[i].time_start;
        player.currentTime = targetTime;
        if (player.paused) {
            playerPlay();
        }
    }

    // --------------------------------------
    // Player show/hide logic etc

    async function playerPlay() {
        try {
            await player.play();
            if (!getPlayerActive()) {
                showPlayerAndFade();
            }
        } catch (error) {
            console.error("Playback failed:", error);
        }
    }

    function onPlayerIxChange() {
        // Should be called when:
        // Mouse has either entered or left player holder area
        // Focus has entered or left thhe player holder and children
        if (getPlayerActive()) {
            clearTimeout(fadeOutId);
            playerHolder.style.opacity = '1.0';
        } else {
            showPlayerAndFade(0);
        }
    }

    function showPlayerAndFade(duration) {
        if (isNaN(duration)) {
            duration = DEFAULT_FADE_DELAY;
        }

        playerHolder.classList.add('no-transition');
        playerHolder.style.opacity = '1.0';
        void playerHolder.offsetHeight; // force reflow
        playerHolder.classList.remove('no-transition');

        clearTimeout(fadeOutId);
        fadeOutId = setTimeout(() => {
            playerHolder.style.opacity = fadeOutValue;
        }, duration);
    }

    function getPlayerActive() {
        return isPlayerFocused || isPlayerHover;
    }

    // ----------------------------------------

    function getCurrentSpan() {
        i = getCurrentSegmentIndex()
        if (i == -1) {
            return null
        }
        id = `segment-${i}`
        span = document.getElementById(id);
        return span;
    }

    /**
     * Returns the index of the segment that spans the current play time, or -1.
     */
    function getCurrentSegmentIndex() {
        const seconds = player.currentTime
        for (let i = 0; i < timedTextSegments.length; i++) {
            segment = timedTextSegments[i];
            if (seconds >= segment.time_start && seconds < segment.time_end) {
                return i
            }
        }
        return -1;
    }



    /**
     * Cycles between [none], value[0], value[1], etc.
     * Returns the value which was set on the dataAttribute
     */
    function cycleRootAttribute(dataAttribute, values) {

        if (!Array.isArray(values) || values.length == 0) {
            console.warning("bad value for `values`");
            return;
        }

        const currentValue = root.getAttribute(dataAttribute);
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
            targetValue = ""
        } else {
            targetValue = values[nextIndex];
        }

        // console.log('currentvalue', currentValue)
        // console.log('currentindex', currentIndex)
        // console.log('nextindex', nextIndex)
        // console.log('targetvalue', targetValue)

        if (targetValue == "") {
            root.removeAttribute(dataAttribute);
        } else {
            root.setAttribute(dataAttribute, targetValue);
        }

        // And persist
        localStorage.setItem(dataAttribute, targetValue);

        return targetValue;
    }

    /**
     * Returns the value which was set, if any
     */
    function setRootAttributeFromLocalStorage(dataAttribute) {
        value = localStorage.getItem(dataAttribute)
        if (value) {
            root.setAttribute(dataAttribute, value)
            return value;
        }
        return null;
    }

    /**
     * Iterates the children of `holder` and sets `display visible`
     * when it has a data attribute which is equal to `value`, else sets `display none`.
     *
     * Rem, when markup is "data-thing", dataAttribute should be "thing" (not "data-thing")
     */
    function setChildVisibleWhen(holder, dataAttribute, targetValue) {

        console.log('xxx', targetValue)

        for (const child of holder.children) {
            child.style.display = "none";
        }

        for (const child of holder.children) {

            value = child.dataset[dataAttribute]

            let isMatch = (value === targetValue);
            isMatch |= (!value || value == "default") && (!targetValue || targetValue == "default");
            if (isMatch) {
                child.style.display = 'revert';
                break;
            }
        }
    }

    window.app = {
        init: init
    };

})();
