class AppUtil {
    
    /**
     * Hardcoded "internal redirect" for when url looks like one of the project's "official"
     * demo sound file urls, which have been moved to a new domain
     */
    static
    transformAudioUrl(url) {

        const previous = "zeropointnine.github.io/tts-audiobook-tool"
        if (!url.includes(previous)) {
            return url
        }

        const newBaseUrl = "https://zeropointnine.github.io/tts-audiobook-tool-sample-output/"

        const o = {
            "waves-chatterbox": "waves-chatterbox.abr.m4a",
            "waves-glm": "waves-glm.abr.m4a",
            "waves-higgs.abr": "waves-higgs.abr.m4a",
            "waves-higgs-different-voice": "waves-higgs-different-voice.abr.m4a",
            "waves-indextts2": "waves-indextts2.abr.m4a",
            "waves-indextts2-plus-emo": "waves-indextts2-plus-emo.abr.m4a",
            "waves-mira": "waves-mira.abr.m4a",
            "waves-oute": "waves-oute.abr.m4a",
            "waves-s1-mini": "waves-s1-mini.abr.m4a",
            "waves-vibevoice-1.5b": "waves-vibevoice-1.5b.abr.m4a",
            "waves-vibevoice-1.5b-lora-klett": "waves-vibevoice-1.5b-lora-klett.abr.m4a",
            
            "waves-vibevoice-1.5b-lora-hsrjl": "waves-vibevoice-1.5b-lora-hsrjl.abr.m4a"
        };

        for (const [substring, newFilename] of Object.entries(o)) {
            if (url.includes(substring)) {
                const newUrl = `${newBaseUrl}/${newFilename}`;
                console.log('internal redirect')
                console.log("old:", url)
                console.log("new:", newUrl);
                return newUrl;
            }
        }
    }
}
