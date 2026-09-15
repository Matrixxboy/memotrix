---
title: "Knowledge graphs and FHIR"
---

# Use case: knowledge graphs, GeoJSON, and FHIR

## FHIR

`.json` with `resourceType` (string) is routed to `FHIRExtractor`.

## GeoJSON

`.geojson` or JSON with `type` `Feature` / `FeatureCollection`.

## Knowledge graphs

Extensions: `.ttl` `.nt` `.nq` `.rdf` `.owl` `.trig` `.jsonld` `.graphml`. JSON-LD (`@graph` or `@context`+`@id`) sniffed from `.json`. Requires `pip install "memotrix[knowledge]"` (`rdflib`) for RDF parses.

These still become **text chunks** for hybrid search. Memotrix is not a SPARQL engine.

## SCORM

`.scorm` or a zip that contains `imsmanifest.xml` is parsed as a learning package. Other zips are rejected.
