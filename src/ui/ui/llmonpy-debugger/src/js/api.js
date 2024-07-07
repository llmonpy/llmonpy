
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

  async getTraceList() {
    const response = await fetch(this.apiUrl + '/get_trace_list');
    const data = await response.json();
    return data;
  }

  async getCompleteTrace(traceId) {
    const url = this.apiUrl + '/get_complete_trace_by_id?trace_id=' + traceId;
    const response = await fetch(url);
    const data = await response.json();
    return data;
  }

  async getTourneyStepNameList() {
    const response = await fetch(this.apiUrl + '/get_tourney_step_name_list');
    const data = await response.json();
    return data;
  }

  async getTourneyResultsForStepName(stepName) {
    const url = this.apiUrl + '/get_tourney_results_for_step_name?step_name=' + stepName;
    const response = await fetch(url);
    const data = await response.json();
    return data;
  }

  async getEventsForStep(stepId) {
    const url = this.apiUrl + '/get_events_for_step?step_id=' + stepId;
    const response = await fetch(url);
    const data = await response.json();
    return data;
  }
}

export class DisplayTrace {
  constructor(trace) {
    this.trace = trace;
    this.traceId = trace.trace_id;
    this.traceName = DisplayTrace.generateDisplayName(trace);
  }

  static generateDisplayName(trace) {
    const lastPeriod = trace.title.lastIndexOf('.');
    let result = trace.title.substring(lastPeriod + 1) + " " + trace.start_time;
    return result;
  }

  static CreateDisplayTraceList(traceList) {
    let result = [];
    for (let trace of traceList) {
      result.push(new DisplayTrace(trace));
    }
    return result;
  }
}

export class DisplayStep {
  constructor(step, parentStep, tourneyResults, componentName) {
    this.step = step;
    this.displayName = step.step_name;
    this.parentStep = parentStep;
    this.tourneyResults = tourneyResults;
    this.componentName = componentName;
    this.children = [];
  }

  addChild(child) {
    this.children.push(child);
  }

  static CreateDisplayStepList(stepList, tourneyResultList, componentMap, defaultComponentName) {
    let result = [];
    let stepMap = new Map();
    // stepList is ordered by step_index, which means we will see parent steps before child steps
    for (let step of stepList) {
      let parentStep = stepMap.get(step.parent_step_id);
      let componentName = defaultComponentName;
      if (step.step_type in componentMap) {
        componentName = componentMap[step.step_type];
      }
      const tourneyResults = DisplayStep.FindTourneyResultsForStepId(step.step_id, tourneyResultList);
      const displayStep = new DisplayStep(step, parentStep, tourneyResults, componentName);
      result.push(displayStep);
      stepMap.set(step.step_id, displayStep);
      if (parentStep != null) {
        parentStep.addChild(displayStep);
      }
    }
    return result;
  }

  static FindTourneyResultsForStepId(stepId, tourneyResultList) {
    let result = null;
    for (let tourneyResult of tourneyResultList) {
      if (tourneyResult.step_id == stepId) {
        result = tourneyResult;
        break;
      }
    }
    return null;
  }
}

export function LLMClientSettingsToString(settings) {
  let result = "";
  if (settings != null) {
    let addSeparator = false;
    result += "{ "
    for (let key in settings) {
      if (addSeparator) {
        result += ", ";
      }
      result += key + ":" + settings[key];
    }
    result += " }";
  }
  return result;
}
