// code by jph
package ch.ethz.idsc.subare.core.util;

import java.io.Serializable;
import java.util.stream.Stream;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionModel;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.TensorRuntimeException;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Max;
import ch.ethz.idsc.tensor.red.Min;

public class DiscreteQsa implements QsaInterface, DiscreteValueFunction, Serializable {
  /** @param stateActionModel
   * @return qsa with q(s, a) == 0 for all state-action pairs */
  public static DiscreteQsa build(StateActionModel stateActionModel) {
    Index index = DiscreteUtils.build(stateActionModel, stateActionModel.states());
    return new DiscreteQsa(index, Array.zeros(index.size()));
  }

  public static DiscreteQsa build(MonteCarloInterface monteCarloInterface, Scalar init) {
    Index index = DiscreteUtils.build(monteCarloInterface, monteCarloInterface.states());
    return new DiscreteQsa(index, Tensors.vector(i -> monteCarloInterface.isTerminal(index.get(i).get(0)) //
        ? RealScalar.ZERO
        : init, index.size()));
  }

  /***************************************************/
  private final Index index;
  private final Tensor values;

  private DiscreteQsa(Index index, Tensor values) {
    if (index.size() != values.length())
      throw TensorRuntimeException.of(index.keys(), values);
    this.index = index;
    this.values = values;
  }

  @Override // from QsaInterface
  public Scalar value(Tensor state, Tensor action) {
    return values.Get(index.of(StateAction.key(state, action)));
  }

  @Override // from QsaInterface
  public void assign(Tensor state, Tensor action, Scalar value) {
    values.set(value, index.of(StateAction.key(state, action)));
  }

  @Override // from QsaInterface
  public DiscreteQsa copy() {
    return new DiscreteQsa(index, values.copy());
  }

  /**************************************************/
  @Override // from DiscreteValueFunction
  public Tensor keys() {
    return index.keys();
  }

  @Override // from DiscreteValueFunction
  public Tensor values() {
    return values.unmodifiable();
  }

  @Override // from DiscreteValueFunction
  public DiscreteQsa create(Stream<? extends Tensor> stream) {
    return new DiscreteQsa(index, Tensor.of(stream));
  }

  /**************************************************/
  public Scalar getMin() {
    return (Scalar) values.flatten(-1).reduce(Min::of).get();
  }

  public Scalar getMax() {
    return (Scalar) values.flatten(-1).reduce(Max::of).get();
  }

  public int size() {
    return index.size();
  }
}
