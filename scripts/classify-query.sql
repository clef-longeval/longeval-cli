--
-- Classify boolean queries
-- This misses queries like: "combo meals option and (yearpublished>=1977 and yearpublished<=2025)" but since no " are used it is difficult to identify
ALTER TABLE queries
ADD COLUMN is_boolean BOOLEAN DEFAULT FALSE;

UPDATE queries
SET is_boolean = TRUE
WHERE query_text ~ '(AND|OR|NOT|" or "|" and ")';

--
-- Classify DOIs
ALTER TABLE queries
ADD COLUMN has_doi BOOLEAN DEFAULT FALSE;

UPDATE queries
SET has_doi = TRUE
WHERE query_text ~ '(^|[^0-9])10\.[0-9]{4,9}/[-._;()/:a-zA-Z0-9]+';


--
-- Add word count
ALTER TABLE queries
ADD COLUMN n_terms INTEGER;

UPDATE queries
SET n_terms = array_length(regexp_split_to_array(trim(query_text), '\s+'), 1);

--
-- Classify structured queries
ALTER TABLE queries
ADD COLUMN is_structured BOOLEAN DEFAULT FALSE;

UPDATE queries
SET is_structured = FALSE;

UPDATE queries
SET is_structured = TRUE
-- Field lookup: "yearPublished" "authors" "dataProvider" "documentType" "publishedDate" "updatedDate" "language" "publisher" "fieldOfStudy"
WHERE query_text ~* '(^|[[:space:]]|[^a-zA-Z0-9])(|abstract|acceptedDate|arxivId|authors|citationCount|contributors|createdDate|dataProviders|depositedDate|documentType|doi|downloadUrl|fieldOfStudy|fullText|identifiers|journals|links|magId|oaiIds|outputs|publishedDate|publisher|pubmedId|references|sourceFulltextUrls|title|updatedDate|yearPublished|language|_exists_|id|yearpublished|updateddate):';



-- Add a was clicked column
ALTER TABLE queries 
ADD COLUMN was_clicked BOOLEAN DEFAULT FALSE;


UPDATE queries q
SET was_clicked = true
FROM search_logs_normalized s
JOIN tracker_logs t ON s.search_id = t.search_id
WHERE q.query_id = s.query_id;
