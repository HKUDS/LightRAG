#!/bin/bash

NAMESPACE=rag
helm uninstall lightrag --namespace $NAMESPACE
