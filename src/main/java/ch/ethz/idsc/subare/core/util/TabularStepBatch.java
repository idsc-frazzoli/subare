// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.SampleModel;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.adapter.StepAdapter;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

class TabularStepBatch {
  private final SampleModel sampleModel;
  private final List<Tensor> list;
  private int index = 0;

  public TabularStepBatch(DiscreteModel discreteModel, SampleModel sampleModel) {
    this.sampleModel = sampleModel;
    Index index = DiscreteUtils.build(discreteModel, discreteModel.states());
    list = index.keys().flatten(0).collect(Collectors.toList());
    Collections.shuffle(list);
  }

  public boolean hasNext() {
    return index < list.size();
  }

  public StepInterface next() {
    Tensor key = list.get(index);
    ++index;
    return step(key.get(0), key.get(1));
  }

  private StepInterface step(Tensor state, Tensor action) {
    Tensor next = sampleModel.move(state, action);
    Scalar reward = sampleModel.reward(state, action, next);
    return new StepAdapter(state, action, reward, next);
  }
}
