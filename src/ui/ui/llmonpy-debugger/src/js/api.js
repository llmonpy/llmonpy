
export let API = null;

export function InitLLMonPyScopeAPI(apiUrl) {
  API = new LLMonPyScopeAPI(apiUrl);
}

export class LLMonPyScopeAPI {
  constructor(apiUrl) {
    this.apiUrl = apiUrl;
    console.log("apiUrl:" + apiUrl);
    this.testApPI();
  }

  async testApPI() {
    const response = await fetch(this.apiUrl + '/hello_world');
    const data = await response.json();
    console.log(data);
  }

}
