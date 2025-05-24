(function() {

    let rawText = "";
    let timedTextSegments = [];

    let loadFileInput = null;
    let fileNameDiv = null;
    let audioPlayer = null;
    let textHolder = null;
    let themeButton = null;
    let loadLocalButtonLabel = null;
    let loadUrlInput = null;

    let selectedSpan = null;
    let intervalId = -1;

    function init() {

        loadFileInput = document.getElementById('loadFileInput');
        fileNameDiv = document.getElementById('fileName')
        audioPlayer = document.getElementById('audioPlayer');
        textHolder = document.getElementById('textHolder');
        themeButton = document.getElementById('themeButton');
        loadLocalButtonLabel = document.getElementById("loadLocalButtonLabel");
        loadUrlInput = document.getElementById('loadUrlInput');

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

        audioPlayer.addEventListener('play', function() {
            selectedSpan = null; // ensures scroll to current audio segment
        });

        textHolder.addEventListener('click', (event) => {
            const target = event.target;
            if (target.tagName === 'SPAN' && target.id.startsWith('segment-')) {
                const segmentIndex = parseInt(target.id.split('-')[1]);
                if (!isNaN(segmentIndex) && timedTextSegments[segmentIndex]) {
                    if (selectedSpan && selectedSpan != target) {
                        selectedSpan.classList.remove('highlight');
                    }
                    audioPlayer.currentTime = timedTextSegments[segmentIndex].time_start;
                    audioPlayer.play();
                }
            }
        });

        // Color theme toggle
        const html = document.documentElement;
        themeButton.addEventListener('click', () => {
            if (html.getAttribute('data-theme') === 'dark') {
                html.removeAttribute('data-theme');
                localStorage.setItem('darkMode', 'false');
            } else {
                html.setAttribute('data-theme', 'dark');
                localStorage.setItem('darkMode', 'true');
            }
        });
        if (localStorage.getItem('darkMode') === 'true') {
            html.setAttribute('data-theme', 'dark');
        }

        // When the queryparam is "url", run the function "loadFlacOrMp4" using the value.
        const urlParams = new URLSearchParams(window.location.search);
        const url = urlParams.get('url');
        if (url) {
            loadFlacOrMp4(url);
        }
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
        clearInterval(intervalId)
        audioPlayer.src = null;
        audioPlayer.style.display = "none";
        fileNameDiv.style.display = "none"
        textHolder.style.display = "none";
        selectedSpan = null;

        document.removeEventListener("keydown", onKeyDown);
    }

    function start(fileOrUrl, pRawText, pTimedTextSegments) {

        let file = null;
        let url = null;
        if (typeof fileOrUrl === "string") {
            url = fileOrUrl;
        } else {
            file = fileOrUrl;
        }

        // TODO: deeplink-driven or at least add to history or smth
        // if (url) {
        //     const newUrl = new URL(window.location.href);
        //     newUrl.searchParams.set('url', url);
        //     window.history.pushState({ url: url }, document.title, newUrl.toString());
        // }

        rawText = pRawText
        timedTextSegments = pTimedTextSegments

        fileNameDiv.style.display = "block"
        fileNameDiv.textContent = file ? file.name : url

        populateText()

        audioPlayer.src = file ? URL.createObjectURL(file) : url
        audioPlayer.play();
        audioPlayer.style.display = "block";

        document.addEventListener("keydown", onKeyDown);

        intervalId = setInterval(loop, 50)
    }

    function onKeyDown(event) {
        if (event.target.tagName == "INPUT") {
            return;
        }
        // console.log(event.key);
        switch (event.key) {
            case "Escape":
                if (audioPlayer.paused) {
                    audioPlayer.play();
                } else {
                    audioPlayer.pause();
                }
                event.preventDefault();
                break;
            case "[":
                seekPreviousSegment();
                break;
            case "]":
                seekNextSegment();
                break;

            case "Enter": // falls through
            case " ":
                if (document.activeElement == loadLocalButtonLabel) {
                    loadFileInput.click();
                    event.preventDefault();
                }
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
        audioPlayer.currentTime = targetTime;
        if (audioPlayer.paused) {
            audioPlayer.play();
        }
    }

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
        const seconds = audioPlayer.currentTime
        for (let i = 0; i < timedTextSegments.length; i++) {
            segment = timedTextSegments[i];
            if (seconds >= segment.time_start && seconds < segment.time_end) {
                return i
            }
        }
        return -1;
    }

    window.app = {
        init: init
    };

})();
