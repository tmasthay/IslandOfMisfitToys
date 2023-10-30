function extract_hashes(marker) {
    let elements = document.getElementsByTagName('c-data');
    let filteredElements = Array.from(elements).filter(element => {
        return element.getAttribute('jsdata') && element.getAttribute('jsdata').includes(marker);
    });

    let extractedArray = filteredElements.map(element => {
        let match = element.getAttribute('jsdata').match(new RegExp(`.*${marker};(.*?);.*`));
        return match ? match[1] : null;
    });

    return extractedArray;  // Returns the array of captured groups
}

// Usage
let marker = 'KCtMme';
extract_hashes(marker);
