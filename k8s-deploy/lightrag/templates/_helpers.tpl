{{/*
Application name
*/}}
{{- define "lightrag.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Full application name
*/}}
{{- define "lightrag.fullname" -}}
{{- default .Release.Name .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "lightrag.labels" -}}
app.kubernetes.io/name: {{ include "lightrag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "lightrag.selectorLabels" -}}
app.kubernetes.io/name: {{ include "lightrag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
.env file content
*/}}
{{- define "lightrag.envContent" -}}
{{- $first := true -}}
{{- range $key, $val := .Values.env -}}
{{- if not $first -}}{{- "\n" -}}{{- end -}}
{{- $first = false -}}
{{ $key }}={{ $val }}
{{- end -}}
{{- end -}}

{{/*
Name of the StatefulSet's governing headless Service.

A Service name is a DNS label, capped at 63 characters, so the base name is
truncated to 54 to leave room for the suffix -- appending it to a full-length
lightrag.fullname would render a name the API server rejects.
*/}}
{{- define "lightrag.headlessServiceName" -}}
{{- printf "%s-headless" (include "lightrag.fullname" . | trunc 54 | trimSuffix "-") -}}
{{- end -}}

{{/*
Workload kind: Deployment (default) or StatefulSet.
StatefulSet exists for ordered startup (podManagementPolicy: OrderedReady):
each pod must be Ready -- which for LightRAG means /health returned 200, so
storage initialization and migration have finished -- before the next pod is
created. See K8S-README.md "Replica Count".
*/}}
{{- define "lightrag.workloadKind" -}}
{{- $workload := default (dict) .Values.workload -}}
{{- $kind := default "Deployment" $workload.kind -}}
{{- if not (has $kind (list "Deployment" "StatefulSet")) -}}
{{- fail (printf "workload.kind must be \"Deployment\" or \"StatefulSet\", got %q" $kind) -}}
{{- end -}}
{{- $kind -}}
{{- end -}}

{{/*
Pod template shared by the Deployment and the StatefulSet, so the two kinds
can never drift apart. Include it under `template:` with nindent 4.

Storage volumes differ by kind on purpose: a Deployment mounts the shared
PVCs from pvc.yaml, while a StatefulSet gets one claim per pod from its
volumeClaimTemplates (a shared ReadWriteOnce claim cannot be mounted by pods
on different nodes, and concurrent writers to one volume corrupt file-based
storage).
*/}}
{{- define "lightrag.podTemplate" -}}
metadata:
  annotations:
    checksum/config: {{ include "lightrag.envContent" . | sha256sum }}
  labels:
    {{- include "lightrag.selectorLabels" . | nindent 4 }}
spec:
  containers:
    - name: {{ .Chart.Name }}
      image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
      imagePullPolicy: IfNotPresent
      ports:
        - name: http
          containerPort: {{ .Values.env.PORT }}
          protocol: TCP
      readinessProbe:
        httpGet:
          path: /health
          port: http
        initialDelaySeconds: 10
        periodSeconds: 5
        timeoutSeconds: 2
        successThreshold: 1
        failureThreshold: 3
      resources:
        {{- toYaml .Values.resources | nindent 8 }}
      volumeMounts:
        - name: rag-storage
          mountPath: /app/data/rag_storage
        - name: inputs
          mountPath: /app/data/inputs
        - name: env-file
          mountPath: /app/.env
          subPath: .env
      {{- $envFrom := default (dict) .Values.envFrom }}
      {{- $envFromEntries := list }}
      {{- range (default (list) (index $envFrom "secrets")) }}
      {{- $envFromEntries = append $envFromEntries (dict "secretRef" (dict "name" .name)) }}
      {{- end }}
      {{- range (default (list) (index $envFrom "configmaps")) }}
      {{- $envFromEntries = append $envFromEntries (dict "configMapRef" (dict "name" .name)) }}
      {{- end }}
      {{- if gt (len $envFromEntries) 0 }}
      envFrom:
{{- toYaml $envFromEntries | nindent 8 }}
      {{- end }}
  {{- with .Values.image.imagePullSecrets }}
  imagePullSecrets:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  volumes:
    - name: env-file
      secret:
        secretName: {{ include "lightrag.fullname" . }}-env
    {{- if not .Values.persistence.enabled }}
    - name: rag-storage
      emptyDir: {}
    - name: inputs
      emptyDir: {}
    {{- else if eq (include "lightrag.workloadKind" .) "Deployment" }}
    - name: rag-storage
      persistentVolumeClaim:
        claimName: {{ include "lightrag.fullname" . }}-rag-storage
    - name: inputs
      persistentVolumeClaim:
        claimName: {{ include "lightrag.fullname" . }}-inputs
    {{- end }}
{{- end -}}
