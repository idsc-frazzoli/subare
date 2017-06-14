// code by jph
package ch.ethz.idsc.subare.core.util;

import java.io.Serializable;
import java.util.function.Function;
import java.util.stream.Stream;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Max;
import ch.ethz.idsc.tensor.red.Min;

public class DiscreteQsa implements QsaInterface, TensorValuesInterface, Serializable {
  public static DiscreteQsa build(DiscreteModel discreteModel) {
    Index index = DiscreteUtils.build(discreteModel, discreteModel.states());
    return new DiscreteQsa(index, Array.zeros(index.size()));
  }

  /** @param state
   * @param action
   * @return */
  // TODO move function since, it's not just used in qsa ops
  public static Tensor createKey(Tensor state, Tensor action) {
    return Tensors.of(state, action);
  }

  public static Tensor createKey(StepInterface stepInterface) {
    return Tensors.of(stepInterface.prevState(), stepInterface.action());
  }

  final Index index;
  final Tensor values;

  private DiscreteQsa(Index index, Tensor values) {
    if (index.size() != values.length())
      throw new RuntimeException();
    this.index = index;
    this.values = values;
  }

  @Override
  public Scalar value(Tensor state, Tensor action) {
    return values.Get(index.of(createKey(state, action)));
  }

  @Override
  public void assign(Tensor state, Tensor action, Scalar value) {
    values.set(value, index.of(createKey(state, action)));
  }

  public void print() {
    print(Function.identity());
  }

  @Override
  public DiscreteQsa copy() {
    return new DiscreteQsa(index, values.copy());
  }

  public void print(Function<Scalar, Scalar> ROUND) {
    for (Tensor key : index.keys()) {
      Scalar value = values.Get(index.of(key));
      System.out.println(key + " " + value.map(ROUND));
    }
  }

  /**************************************************/
  @Override
  public Tensor keys() {
    return index.keys();
  }

  @Override
  public Tensor values() {
    return values.unmodifiable();
  }

  @Override
  public DiscreteQsa create(Stream<? extends Tensor> stream) {
    return new DiscreteQsa(index, Tensor.of(stream));
  }

  /**************************************************/
  public Scalar getMin() {
    return values.flatten(-1).map(Scalar.class::cast).reduce(Min::of).get();
  }

  public Scalar getMax() {
    return values.flatten(-1).map(Scalar.class::cast).reduce(Max::of).get();
  }

  public int size() {
    return index.size();
  }
}
