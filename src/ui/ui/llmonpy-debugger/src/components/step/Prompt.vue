<template>
  <v-card>
    <v-card-subtitle>{{displayStep.step.step_type}}</v-card-subtitle>
    <v-card-title @click="showChildren = !showChildren">{{displayStep.displayName}}</v-card-title>
  </v-card>
</template>

<script>


export default {
  name: 'Prompt',
  components: {},
  watch: {
    displayStep: function (newVal, oldVal) {
      this.setup()
    }
  },
  props: ['displayStep'],
  data: () => ({
    showChildren: false,
  }),
  methods: {
    setup() {
      console.log("Prompt: " + this.displayStep.step.step_id)
    }
  },
  created() {
    this.setup()
  },
  computed: {},

}
</script>
