// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Arrays;
import java.util.List;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.SampleModel;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;

public enum TabularSteps {
  ;
  // ---
  public static void batch(DiscreteModel discreteModel, SampleModel sampleModel, StepDigest... stepDigest) {
    List<StepDigest> list = Arrays.asList(stepDigest);
    TabularStepBatch tabularStepBatch = new TabularStepBatch(discreteModel, sampleModel);
    while (tabularStepBatch.hasNext()) {
      StepInterface stepInterface = tabularStepBatch.next();
      list.stream().parallel().forEach(_stepDigest -> _stepDigest.digest(stepInterface));
    }
  }
}
