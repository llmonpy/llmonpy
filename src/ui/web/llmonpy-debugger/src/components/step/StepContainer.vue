<template>
  <v-card density="compact" hover>
    <v-card-item>
      <v-card-title  >
        <v-row>
          <v-col cols="8" >
            <v-card-subtitle>
              <span class="font-weight-bold">Time:</span> <span>{{durationInSecondsString}} seconds</span>
              <span class="font-weight-bold ml-2">Cost:</span> <span>${{cost}}</span>
              <template v-if="modelName != null">
                <span class="font-weight-bold ml-2">Model: </span>
                <span>
                  {{modelName}}
                  {{settingsString}}
              </span>
              </template>
              <span class="font-weight-bold ml-2">Type:</span> <span>{{displayStep.step.step_type}}</span>
            </v-card-subtitle>
            {{displayStep.displayName}}
          </v-col>
          <v-col cols="4" class="d-flex justify-end">
            <v-btn-toggle multiple v-model="showList" variant="outlined" color="primary"  density="compact">
              <v-btn v-for="item in showOptionList"
                     :key="item"
                     :value="item"
                     class="mx-1"
                     size="x-small"
                     rounded="lg"
                     density="compact">
                  {{item}}
              </v-btn>
            </v-btn-toggle>
          </v-col>
        </v-row>
      </v-card-title>
    </v-card-item>
    <v-card-text>
      <v-row v-if="showErrors">
        <v-col>
          <v-card variant="tonal" density="compact" color="red-accent-4">
            <v-card-title style="font-size: 0.9rem">
              Errors
            </v-card-title>
            <v-card-text>
              <v-list density="compact" :items="stepErrors">
              </v-list>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>
      <v-row v-if="showReport">
        <v-col>
          <v-card variant="tonal" density="compact" color="primary">
            <v-card-title style="font-size: 0.9rem">
              Cost Per Victory Report
            </v-card-title>
            <v-card-text>
              <v-list density="compact">
                <v-list-item v-for="report in stepModelReportList" :key="displayStep.step.step_id + report.fullName">
                  <span>{{report.fullName}}: </span>
                  <span class="font-weight-bold">${{report.getCostPerVictoryString()}}</span>
                  <span>&nbsp; Victories: </span>
                  <span>{{report.victoryCount}}</span>
                </v-list-item>
              </v-list>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>
      <v-row v-if="showInput">
        <v-col>
          <v-card variant="tonal" density="compact" color="primary">
            <v-card-title style="font-size: 0.9rem">
              Input
            </v-card-title>
            <v-card-text>
              <vue-json-pretty class="ml-2" :data="stepInput"/>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>
      <v-row v-if="showOutput">
        <v-col>
          <v-card variant="tonal" density="compact" color="primary">
            <v-card-title style="font-size: 0.9rem">
              Output
            </v-card-title>
            <v-card-text>
              <vue-json-pretty class="ml-2" :data="stepOutput"/>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>
      <v-row v-if="showLogs">
        <v-col>
            <v-card variant="tonal" density="compact" color="primary">
            <v-card-title style="font-size: 0.9rem">
              Logs
            </v-card-title>
            <v-card-text>
              <template v-if="stepEvents.length > 0">
                <step-log
                    :event-list="stepEvents">
                </step-log>
              </template>
              <v-alert v-show="stepEvents.length == 0" type="info" class="mx-4 mt-1" density="compact">
                Nothing Logged
              </v-alert>
            </v-card-text>
            </v-card>
        </v-col>
      </v-row>
    </v-card-text>
    <template v-if="showSteps">
      <template v-for="(child) in displayStep.children"
                :key="child.step.step_id">
        <step-container
          :display-step="child"
          class="mx-4 my-4"></step-container>
      </template>
    </template>
  </v-card>
</template>

<script>

import Prompt from "@/components/step/Prompt.vue";
import {API, CalculateDuration, LLMClientSettingsToString, ModelReport} from "@/js/api";
import VueJsonPretty from "vue-json-pretty";
import 'vue-json-pretty/lib/styles.css';
import StepLog from "@/components/step/StepLog.vue";

const SHOW_INPUT = "Input";
const SHOW_OUTPUT = "Output";
const SHOW_LOGS = "Logs";
const SHOW_ERRORS = "Errors";
const SHOW_STEPS = "Steps";
const SHOW_REPORT = "Report";

export default {
  name: 'StepContainer',
  components: {StepLog, Prompt, VueJsonPretty},
  watch: {
    displayStep: function (newVal, oldVal) {
      this.setup()
    },
    showList: function (newVal, oldVal) {
      this.showReport = this.showList.includes(SHOW_REPORT);
      this.showInput = this.showList.includes(SHOW_INPUT);
      this.showOutput = this.showList.includes(SHOW_OUTPUT);
      this.showLogs = this.showList.includes(SHOW_LOGS);
      this.showErrors = this.showList.includes(SHOW_ERRORS);
      this.showSteps = this.showList.includes(SHOW_STEPS);
      if ( this.showLogs && this.logsLoaded == false ) {
        this.stepEvents = this.loadLogs();
      }
    }
  },
  props: ['displayStep'],
  data: () => ({
    showReport: false,
    showInput: false,
    showOutput: false,
    showLogs: false,
    showErrors: false,
    showSteps: false,
    showChildren: false,
    cost: 0.0,
    modelName: null,
    settingsString: null,
    durationInSecondsString: "0.0",
    stepOutput: null,
    stepInput: null,
    stepErrors: null,
    stepEvents: [],
    stepModelReportList: null,
    logsLoaded: false,
    loadingLogs: false,
    showList:[],
    showOptionList: [SHOW_REPORT, SHOW_INPUT, SHOW_OUTPUT, SHOW_LOGS, SHOW_ERRORS, SHOW_STEPS],
  }),
  methods: {
    setup() {
      console.log("StepContainer: " + this.displayStep.step.step_id)
      this.cost = new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 5
      }).format(this.displayStep.step.cost);
      const durationInSeconds = CalculateDuration(this.displayStep.step.start_time, this.displayStep.step.end_time);
      this.durationInSecondsString = new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 3,
        maximumFractionDigits: 3
      }).format(durationInSeconds);
      this.stepOutput = this.displayStep.step.output_dict;
      this.stepInput = this.displayStep.step.input_dict;
      this.stepErrors = this.displayStep.step.error_list;
      if (this.displayStep.step.llm_model_info != null) {
        this.modelName = this.displayStep.step.llm_model_info.model_name;
        this.settingsString = LLMClientSettingsToString(this.displayStep.step.llm_model_info.client_settings_dict);
      } else {
        this.modelName = null;
        this.settingsString = null;
      }
      this.stepModelReportList = ModelReport.GenerateModelReportList(this.displayStep.children)
      if ( this.stepModelReportList == null ) {
        this.showOptionList = this.showOptionList.filter(item => item != SHOW_REPORT);
      }
      if ( this.displayStep.children == null || this.displayStep.children.length == 0 ) {
        this.showOptionList = this.showOptionList.filter(item => item != SHOW_STEPS);
      }
      if ( this.displayStep.step.input_dict == null || Object.keys(this.displayStep.step.input_dict).length == 0 ) {
        this.showOptionList = this.showOptionList.filter(item => item != SHOW_INPUT);
      }
      if ( this.displayStep.step.error_list == null || this.displayStep.step.error_list.length == 0 ) {
        this.showOptionList = this.showOptionList.filter(item => item != SHOW_ERRORS);
      }
    },
    async loadLogs() {
      this.loadingLogs = true;
      try {
        const stepEvents = await API.getEventsForStep(this.displayStep.step.step_id);
        this.logsLoaded = true;
        this.stepEvents = stepEvents;
      } finally {
        this.loadingLogs = false;
      }
    }
  },
  created() {
    this.setup()
  },
  computed: {},

}
</script>
