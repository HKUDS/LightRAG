{{/*
Expand the name of the chart.
*/}}
{{- define "lightrag-minimal.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "lightrag-minimal.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "lightrag-minimal.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "lightrag-minimal.labels" -}}
helm.sh/chart: {{ include "lightrag-minimal.chart" . }}
{{ include "lightrag-minimal.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "lightrag-minimal.selectorLabels" -}}
app.kubernetes.io/name: {{ include "lightrag-minimal.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "lightrag-minimal.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "lightrag-minimal.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the secret
*/}}
{{- define "lightrag-minimal.secretName" -}}
{{- printf "%s-secrets" (include "lightrag-minimal.fullname" .) }}
{{- end }}

{{/*
PostgreSQL connection string
*/}}
{{- define "lightrag-minimal.postgresqlHost" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" (include "lightrag-minimal.fullname" .) }}
{{- else }}
{{- .Values.env.POSTGRES_HOST }}
{{- end }}
{{- end }}