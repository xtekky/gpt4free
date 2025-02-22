import PhotoSwipeLightbox from "https://cdn.jsdelivr.net/npm/photoswipe@5.3.8/dist/photoswipe-lightbox.esm.min.js";
import PhotoSwipeVideoPlugin from "https://cdn.jsdelivr.net/gh/dimsemenov/photoswipe-video-plugin@5e32d6589df53df2887900bcd55267d72aee57a6/dist/photoswipe-video-plugin.esm.min.js";
import PhotoSwipeAutoHideUI from "https://cdn.jsdelivr.net/gh/arnowelzel/photoswipe-auto-hide-ui@1.0.1/photoswipe-auto-hide-ui.esm.min.js";
import PhotoSwipeSlideshow from "https://cdn.jsdelivr.net/gh/dpet23/photoswipe-slideshow@v2.0.0/photoswipe-slideshow.esm.min.js";

const lightbox = new PhotoSwipeLightbox({
    gallery: '#messages',
    children: 'a:has(img)',
    initialZoomLevel: 'fill',
    secondaryZoomLevel: 1,
    maxZoomLevel: 2,
    allowPanToNext: true,
    doubleTapAction: 'close',
    pswpModule: () => import('https://cdn.jsdelivr.net/npm/photoswipe'),
});
lightbox.addFilter('itemData', (itemData, index) => {
    const img = itemData.element.querySelector('img');
    itemData.width = img.naturalWidth || 1024;
    itemData.height = img.naturalHeight || 1024;
    return itemData;
});
lightbox.on('uiRegister', function() {
    lightbox.pswp.ui.registerElement({
        name: 'custom-caption',
        order: 9,
        isButton: false,
        appendTo: 'root',
        html: 'Caption text',
        onInit: (el, pswp) => {
            lightbox.pswp.on('change', () => {
                const currSlideElement = lightbox.pswp.currSlide.data.element;
                if (currSlideElement) {
                    const img = currSlideElement.querySelector('img');
                    const download = document.createElement("a");
                    download.setAttribute("href", img.getAttribute('src'));
                    let extension = img.getAttribute('src').includes(".webp") ? ".webp" : ".jpg";
                    download.setAttribute("download", `${img.getAttribute('alt')} ${lightbox.pswp.currSlide.index}${extension}`);
                    download.style.float = "right";
                    download.innerHTML = '<i class="fa-solid fa-download"></i>';
                    let span = document.createElement("span");
                    span.innerText = img.getAttribute('alt');
                    el.innerHTML = '';
                    el.appendChild(download);
                    el.appendChild(span);
                }
            });
        }
    });
});
// Add a slideshow to the PhotoSwipe gallery.
const _slideshowPlugin = new PhotoSwipeSlideshow(lightbox, {
    defaultDelayMs: 7000,
    restartOnSlideChange: true,
    progressBarPosition: "top",
    autoHideProgressBar: false
});

// Plugin to display video.
const _videoPlugin = new PhotoSwipeVideoPlugin(lightbox, {});

// Hide the PhotoSwipe UI after some time of inactivity.
const _autoHideUI = new PhotoSwipeAutoHideUI(lightbox, {});
lightbox.init();