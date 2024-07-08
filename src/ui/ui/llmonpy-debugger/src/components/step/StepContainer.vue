<template>
  <v-card>
    <v-card-item>
      <v-card-subtitle>
        <span class="font-weight-bold ml-2">Cost:</span> <span>{{displayStep.step.cost}}</span>
        <template v-if="modelName != null">
          <span class="font-weight-bold ml-2">Model: </span>
          <span>
                  {{modelName}}
                  {{settingsString}}
              </span>
        </template>
        <span class="font-weight-bold ml-2">Type:</span> <span>{{displayStep.step.step_type}}</span>
      </v-card-subtitle>
      <v-card-title >{{displayStep.displayName}}</v-card-title>
    </v-card-item>
    <v-card-text>
      <v-row>
        <v-col>
          <v-combobox label="Show" :items="showOptionList" v-model="showList" multiple dense></v-combobox>
          <template v-if="showOutput">
            <h4 class="ml-2 mt-4">Output</h4>
            <vue-json-pretty class="ml-2" :data="stepOutput"/>
          </template>
        </v-col>
      </v-row>
    </v-card-text>
    <template v-if="showSteps">
      <template v-for="(child) in displayStep.children"
                :key="child.step.step_id">
        <step-container
          :display-step="child"
          class="ml-4 my-2"></step-container>
      </template>
    </template>
  </v-card>
</template>

<script>

import Prompt from "@/components/step/Prompt.vue";
import {LLMClientSettingsToString} from "@/js/api";
import VueJsonPretty from "vue-json-pretty";
import 'vue-json-pretty/lib/styles.css';

const SHOW_INPUT = "Input";
const SHOW_OUTPUT = "Output";
const SHOW_LOGS = "Logs";
const SHOW_ERRORS = "Errors";
const SHOW_STEPS = "Steps";

export default {
  name: 'StepContainer',
  components: {Prompt, VueJsonPretty},
  watch: {
    displayStep: function (newVal, oldVal) {
      this.setup()
    },
    showList: function (newVal, oldVal) {
      this.showInput = this.showList.includes(SHOW_INPUT);
      this.showOutput = this.showList.includes(SHOW_OUTPUT);
      this.showLogs = this.showList.includes(SHOW_LOGS);
      this.showErrors = this.showList.includes(SHOW_ERRORS);
      this.showSteps = this.showList.includes(SHOW_STEPS);
    }
  },
  props: ['displayStep'],
  data: () => ({
    showInput: false,
    showOutput: false,
    showLogs: false,
    showErrors: false,
    showSteps: false,
    showChildren: false,
    cost: 0.0,
    modelName: null,
    settingsString: null,
    durationInMilliseconds: "0.0",
    stepOutput: null,
    showList:[],
    showOptionList: [SHOW_INPUT, SHOW_OUTPUT, SHOW_LOGS, SHOW_ERRORS, SHOW_STEPS],
  }),
  methods: {
    setup() {
      console.log("StepContainer: " + this.displayStep.step.step_id)
      this.cost = this.displayStep.step.cost;
      this.stepOutput = this.displayStep.step.output_dict;
      if (this.displayStep.step.llm_model_info != null) {
        this.modelName = this.displayStep.step.llm_model_info.model_name;
        this.settingsString = LLMClientSettingsToString(this.displayStep.step.llm_model_info.client_settings_dict);
      } else {
        this.modelName = null;
        this.settingsString = null;
      }
    this.durationInMilliseconds = this.displayStep.step.durationInMilliseconds;
    }
  },
  created() {
    this.setup()
  },
  computed: {},

}
</script>
