#!/bin/bash

NAMESPACE=rag
helm uninstall lightrag-dev --namespace $NAMESPACE
