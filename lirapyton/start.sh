#!/bin/bash
# Archivo: start.sh

echo "Iniciando app en puerto $PORT..."
uvicorn main:app --host 0.0.0.0 --port $PORT
