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

function constructUrls(hashArray) {
    // Use the map() method to create a new array with URLs constructed from each hash
    let urlArray = hashArray.map(extracted_hash => `https://drive.google.com/file/d/${extracted_hash}/view?usp=drive_link`);
    return urlArray;  // Return the new array of URLs
}

function constructUrlDict(basename, urls) {
    let urlDict = {};  // Create an empty object to hold the key-value pairs
    urls.forEach((url, index) => {
        let key = `${basename}${index + 1}`;
        urlDict[key] = url;
    });
    return urlDict;  // Return the constructed object
}

// Usage
let marker = 'KCtMme';
let datatype = 'data';
// let datatype = 'model'; UNCOMMENT for model data
let hashes = extract_hashes(marker);
let urls = constructUrls(hashes);
let d = constructUrlDict(datatype, urls);
console.log(d)
