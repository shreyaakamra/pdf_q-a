---
title: PDF Q&A System
emoji: 📄
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# PDF Q&A System

A Retrieval-Augmented Generation (RAG) system that answers questions
about uploaded PDF documents using FAISS vector search and Groq API.

## Features
- Upload multiple PDFs at once
- Answers grounded in document content
- Shows which document each answer came from
- Says "I don't know" for out-of-scope questions

## How it works
1. Upload one or more PDF documents
2. Ask questions about their content
3. Get answers with source attribution

## Known Limitations
- Tables in PDFs may extract poorly
- Works best with typed/printed PDFs
- First upload may take 30-60 seconds