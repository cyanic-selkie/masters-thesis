#!/usr/bin/env bash

LANGUAGES=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --root-dir)
      ROOT_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    --language)
      LANGUAGES+=("$2")
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option \"$1\""
      exit 1
      ;;
    *)
      echo "Positional arguments (i.e., \"$1\") not supported."
      exit 1;
      shift # past argument
      ;;
  esac
done

if [ -z "$ROOT_DIR" ]; then
  echo "You must provide a root directory for the data";
  exit 1;
fi

if [ ${#LANGUAGES[@]} -eq 0 ]; then
  echo "You must provide at least one language.";
  exit 1;
fi

mkdir -p "$ROOT_DIR/raw/wikidata"

for language in "${LANGUAGES[@]}"
do
  mkdir -p "$ROOT_DIR/raw/wikipedia/$language"
done

wget -c -P "$ROOT_DIR/raw/wikidata" https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.gz

for language in "${LANGUAGES[@]}"
do
  wget -c -P "$ROOT_DIR/raw/wikipedia/$language" "https://dumps.wikimedia.org/${language}wiki/latest/${language}wiki-latest-page.sql.gz"
  wget -c -P "$ROOT_DIR/raw/wikipedia/$language" "https://dumps.wikimedia.org/${language}wiki/latest/${language}wiki-latest-page_props.sql.gz"
  wget -c -P "$ROOT_DIR/raw/wikipedia/$language" "https://dumps.wikimedia.org/${language}wiki/latest/${language}wiki-latest-redirect.sql.gz"
  wget -c -P "$ROOT_DIR/raw/wikipedia/$language" "https://dumps.wikimedia.org/${language}wiki/latest/${language}wiki-latest-pages-articles.xml.bz2"
done

for language in "${LANGUAGES[@]}"
do
  wget -c -P "$ROOT_DIR/raw/wikipedia/$language" "https://dumps.wikimedia.org/${language}wiki/latest/${language}wiki-latest-page.sql.gz"
  wget -c -P "$ROOT_DIR/raw/wikipedia/$language" "https://dumps.wikimedia.org/${language}wiki/latest/${language}wiki-latest-page_props.sql.gz"
  wget -c -P "$ROOT_DIR/raw/wikipedia/$language" "https://dumps.wikimedia.org/${language}wiki/latest/${language}wiki-latest-redirect.sql.gz"
  wget -c -P "$ROOT_DIR/raw/wikipedia/$language" "https://dumps.wikimedia.org/${language}wiki/latest/${language}wiki-latest-pages-articles.xml.bz2"
done

for language in "${LANGUAGES[@]}"
do
  gzip -dc "$ROOT_DIR/raw/wikipedia/${language}/${language}wiki-latest-page.sql.gz" > "$ROOT_DIR/raw/wikipedia/${language}/${language}wiki-latest-page.sql"
  gzip -dc "$ROOT_DIR/raw/wikipedia/${language}/${language}wiki-latest-page_props.sql.gz" > "$ROOT_DIR/raw/wikipedia/${language}/${language}wiki-latest-page_props.sql"
  gzip -dc "$ROOT_DIR/raw/wikipedia/${language}/${language}wiki-latest-redirect.sql.gz" > "$ROOT_DIR/raw/wikipedia/${language}/${language}wiki-latest-redirect.sql"
done