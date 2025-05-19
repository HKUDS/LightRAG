#!/bin/bash

print_title() {
  echo "============================================"
  echo "$1"
  echo "============================================"
}

print_success() {
  echo "✅ $1"
}

print_error() {
  echo "❌ $1"
}

print_warning() {
  echo "⚠️ $1"
}

print_info() {
  echo "🔹 $1"
}

print() {
  echo "$1"
}

# Check dependencies
check_dependencies(){
  print "Checking dependencies..."
  command -v kubectl >/dev/null 2>&1 || { print "Error: kubectl command not found"; exit 1; }
  command -v helm >/dev/null 2>&1 || { print "Error: helm command not found"; exit 1; }

  # Check if Kubernetes is available
  print "Checking if Kubernetes is available..."
  kubectl cluster-info &>/dev/null
  if [ $? -ne 0 ]; then
      print "Error: Kubernetes cluster is not accessible. Please ensure you have proper access to a Kubernetes cluster."
      exit 1
  fi
  print_success "Kubernetes cluster is accessible."
}
