const { read } = require('feed-reader')

async function getEntryDetails (arxivId) {
    const { entries } = await read(`https://export.arxiv.org/api/query?id_list=${arxivId}`, {
        descriptionMaxLen: Number.MAX_VALUE,
        getExtraEntryFields: (feedEntry) => {
            const published = feedEntry.published

            const extraFields = {
                authors: Array.isArray(feedEntry.author) ? feedEntry.author.map(a => a.name) : [feedEntry.author.name],
            };

            if (published) {
                const publishedDate = new Date(published);
                extraFields.month = publishedDate.getMonth() + 1;
                extraFields.year = publishedDate.getFullYear();
            }

            if ("arxiv:doi" in feedEntry) {
                extraFields.doi = feedEntry["arxiv:doi"]["#text"]
            }

            return extraFields
        }
    });

    return entries[0]
}

function atom2bib (entry) {
    const { title, link, description, authors } = entry;
    /*
    @article{2022,
        title={CenterCLIP},
        url={http://dx.doi.org/10.1145/3477495.3531950},
        DOI={10.1145/3477495.3531950},
        journal={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
        publisher={ACM},
        author={Zhao, Shuai and Zhu, Linchao and Wang, Xiaohan and Yang, Yi},
        year={2022},
        month={Jul}
    }
    */

    const extra = [
        entry.month && `month={${entry.month}}`,
        entry.year && `year={${entry.year}}`,
        entry.doi && `DOI={${entry.doi}}`,
    ].filter(e => !!e)

    const fields = [
        `title={${title}}`,
        `url={${link}}`,
        `author={${authors.join(" and ")}}`,
        `abstract={${description}}`,
        ...extra
    ]

    const articleTitle = [
        entry.authors[0].replaceAll(" ", "").toLowerCase(),
        entry.year,
        title.split(" ")[0].toLowerCase()
    ].filter(e => !!e);


    return `@article{{${articleTitle.join("").replaceAll(/[^a-z0-9]/gi, "")}},
    ${fields.join(",\n    ")}
}`
}

getEntryDetails(process.argv.slice(2)[0]).then(e => console.log(atom2bib(e)))
