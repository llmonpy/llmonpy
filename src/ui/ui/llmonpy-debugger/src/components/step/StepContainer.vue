<template>
  <v-card>
    <v-card-item>
      <v-card-subtitle>{{displayStep.step.step_type}}</v-card-subtitle>
      <v-card-title @click="showChildren = !showChildren">{{displayStep.displayName}}</v-card-title>
    </v-card-item>
    <v-card-text>
      <v-row>
        <v-col>
          <span class="font-weight-bold ml-2">Cost:</span> <span>{{displayStep.step.cost}}</span>
          <template v-if="modelName != null">
            <span class="font-weight-bold ml-2">Model: </span>
              <span>
                  {{modelName}}
                  {{settingsString}}
              </span>
          </template>
          <template v-if="stepOutput != null">
            <h4 class="ml-2 mt-4">Output</h4>
            <vue-json-pretty class="ml-2" :data="stepOutput"/>
          </template>
        </v-col>
      </v-row>
    </v-card-text>
    <template v-if="showChildren">
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

export default {
  name: 'StepContainer',
  components: {Prompt, VueJsonPretty},
  watch: {
    displayStep: function (newVal, oldVal) {
      this.setup()
    }
  },
  props: ['displayStep'],
  data: () => ({
    showChildren: true,
    cost: 0.0,
    modelName: null,
    settingsString: null,
    durationInMilliseconds: "0.0",
    stepOutput: null,
  }),
  methods: {
    setup() {
      console.log("StepContainer: " + this.displayStep.step.step_id)
      this.cost = this.displayStep.step.cost;
      this.stepOutput = this.displayStep.step.output_dict;
      if (this.displayStep.step.llm_client_info != null) {
        this.modelName = this.displayStep.step.llm_client_info.client_name;
        this.settingsString = LLMClientSettingsToString(this.displayStep.step.llm_client_info.client_settings_dict);
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
