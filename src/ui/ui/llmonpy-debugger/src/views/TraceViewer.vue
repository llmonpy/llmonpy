<template>
  <v-card title="Trace List">
    <v-card-text>
      <v-select
        v-model="currentTraceId"
        :items="traceList"
        item-title="traceName"
        item-value="traceId"
        label="Select Trace"
      ></v-select>
      <v-divider thickness="5" color="primary"></v-divider>
      <v-row v-if="rootDisplayStep != null">
        <v-col>
          <step-container
            :display-step="rootDisplayStep"
            class="my-2"></step-container>
        </v-col>
      </v-row>
    </v-card-text>
  </v-card>
</template>

<script>


import {API, DisplayStep, DisplayTrace} from "@/js/api";
import Prompt from "@/components/step/Prompt.vue";
import StepContainer from "@/components/step/StepContainer.vue";

export default {
  name: 'TraceViewer',
  components: {StepContainer, Prompt},
  props: [],
  data: () => ({
    currentTraceId: null,
    currentTrace:null,
    displayStepList: [],
    rootDisplayStep: null,
    traceList: [],
    componentMap: {},
  }),
  watch: {
    currentTraceId: function (newVal, oldVal) {
      console.log("currentTraceId: " + newVal);
      this.setupCurrentTrace()
    }
  },
  methods: {
    async setup() {
      const rawTraceList = await API.getTraceList();
      this.traceList = DisplayTrace.CreateDisplayTraceList(rawTraceList);
      if (this.traceList.length > 0 && this.currentTraceId === null) {
        this.currentTraceId = this.traceList[0].traceId;
      }
    },
    async setupCurrentTrace() {
      if (this.currentTraceId != null) {
        this.currentTrace = await API.getCompleteTrace(this.currentTraceId);
        console.log("got trace " + this.currentTrace.trace_info.trace_id);
        this.displayStepList = DisplayStep.CreateDisplayStepList(this.currentTrace.step_list,
                    this.currentTrace.tourney_result_list, this.componentMap, "Prompt");
        this.rootDisplayStep = this.displayStepList[0];
      } else {
        this.currentTrace = null;
      }
    }
  },
  created() {
    this.setup();
  },
  computed: {},

}
</script>
