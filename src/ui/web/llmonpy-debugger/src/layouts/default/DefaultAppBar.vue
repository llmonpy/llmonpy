<template>
  <v-app-bar id="myp-app-bar" app color="primary" dense >
    <v-app-bar-nav-icon @click="drawer = !drawer"/>
    <span class="text-h5 pr-2 py-5">{{ name }}</span>
    <v-spacer/>
    <span class="text-h6 pa-5">7/24/24-1:{{ env }}</span>
  </v-app-bar>
  <v-navigation-drawer v-model="drawer" temporary>
    <v-list>
      <v-list-item><router-link to="/">Trace</router-link></v-list-item>
      <v-list-item><router-link to="/qbawa">Training Data</router-link></v-list-item>
    </v-list>
  </v-navigation-drawer>
</template>

<script>

import {API} from "@/js/api";

export default {
  name: "DefaultAppBar",
  data: () => ({
    drawer: false
  }),
  methods: {
    handleBackClick() {
      this.$router.go(-1);
    }
  },
  computed: {
    name() {
      let result = this.$route.name;
      return result;
    },
    env() {
      return "Development"
    },
    isMobile() {
      return false
    }
  }
};
</script>

