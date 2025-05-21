(function() {

    let rawText = ""
    let timedTextSegments = []

    let flacFileInput = null
    let audioPlayer = null
    let textHolder = null
    let toggleButton = null

    let selectedSpan = null;
    let intervalId = -1

    function init() {
        flacFileInput = document.getElementById('flacFileInput');
        audioPlayer = document.getElementById('audioPlayer');
        textHolder = document.getElementById('textHolder');
        toggleButton = document.getElementById('toggleButton');

        flacFileInput.addEventListener('change', async () => {
            clear()
            const file = flacFileInput.files[0];
            if (!file) {
                return
            }
            result = await loadMetadataFromAppFlac(file)
            if (!result) {
                alert("No tts-audiobook-tool metadata found")
                return
            }
            initFile(file, result["raw_text"], result["text_segments"])
        });

        audioPlayer.addEventListener('play', function() {
            selectedSpan = null; // force scroll to current segment
          });


        // Color theme related
        const html = document.documentElement;
        toggleButton.addEventListener('click', () => {
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
    }

    function clear() {
        clearInterval(intervalId)
        audioPlayer.src = null;
        audioPlayer.style.display = "none";
        textHolder.style.display = "none";
        selectedSpan = null
    }

    function initFile(file, pRawText, pTimedTextSegments) {

        rawText = pRawText
        console.log(rawText)
        timedTextSegments = pTimedTextSegments
        console.log(timedTextSegments)
        populateText()

        audioPlayer.src = URL.createObjectURL(file);
        audioPlayer.play();
        audioPlayer.style.display = "block";

        intervalId = setInterval(loop, 50)
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

        const seconds = audioPlayer.currentTime

        let span = null;
        for (let i = 0; i < timedTextSegments.length; i++) {
            const segment = timedTextSegments[i];
            if (seconds >= segment.time_start && seconds < segment.time_end) {
                span = document.getElementById(`segment-${i}`);
                break; // Found the active segment
            }
        }

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

    window.app = {
        init: init
    };

})();
