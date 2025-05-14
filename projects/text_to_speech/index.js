const worker = {}
if (!worker.current) {
    // Create the worker if it does not yet exist.
    worker.current = new Worker(new URL('./worker.js', import.meta.url), {
        type: 'module'
    });
}

window.doSpeech = false;

const onMessageReceived = (e) => {
    switch (e.data.status) {
        case 'error':
            window.onSpeechResponse(null); 
            window.doSpeech = false;
        break;
        case 'complete':
            const blobUrl = URL.createObjectURL(e.data.output);
            window.onSpeechResponse(blobUrl); 
            window.doSpeech = false;
        break;
    }
};
worker.current.addEventListener('message', onMessageReceived);

import { DEFAULT_SPEAKER, SPEAKERS } from './constants';

const handleGenerateSpeech = (text, speaker_id=DEFAULT_SPEAKER) => {
    window.doSpeech = true;
    worker.current.postMessage({
        text,
        speaker_id: speaker_id,
    });
};

window.SPEAKERS = SPEAKERS;
window.handleGenerateSpeech = handleGenerateSpeech;
window.onSpeechResponse = (url) => console.log(url);