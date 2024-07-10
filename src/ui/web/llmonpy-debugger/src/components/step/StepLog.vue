<template>
  <template v-for="event in eventList">
    <v-card variant="flat">
      <v-card-text>
        <template v-if="event.message">
          <span>{{ event.message }}</span>
        </template>
        <template v-if="event.exception_message">
          <div>
            <span class="font-weight-bold">Exception: </span>
            <span>{{ event.exception_message }}</span>
          </div>
          <div>
            <span class="font-weight-bold">Stack: </span>
            <span>{{ event.stack_trace }}</span>
          </div>
        </template>
        <template v-if="event.prompt_template">
          <span class="font-weight-bold">Prompt Template: </span>
          <span>{{ event.prompt_template }}</span>
        </template>
        <template v-if="event.prompt_text">
          <div>
            <span class="font-weight-bold">Prompt Text: </span>
            <span>{{ event.prompt_text }}</span>
          </div>
          <div class="mt-2">
            <span class="font-weight-bold">Response Text: </span>
            <span>{{ event.response_text }}</span>
          </div>
        </template>
      </v-card-text>
    </v-card>
  </template>
</template>

<script>


export default {
  name: 'StepLog',
  components: {},
  props: ['eventList'],
  data: () => ({}),
  watch: {
    eventList: function (newVal, oldVal) {
      this.setup();
    }
  },
  methods: {
    setup() {
      console.log("StepLog setup");
      console.log(this.eventList);
    }
  },
  created() {
    this.setup();
  },
  computed: {},

}
</script>
