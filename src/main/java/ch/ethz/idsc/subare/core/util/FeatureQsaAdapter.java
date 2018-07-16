// code by fluric
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** Useful to get Qsa values without the need to create all the state-action pair values beforehand */
public class FeatureQsaAdapter implements QsaInterface {
  private final Tensor w;
  private final FeatureMapper featureMapper;

  public FeatureQsaAdapter(Tensor w, FeatureMapper featureMapper) {
    this.w = w;
    this.featureMapper = featureMapper;
  }

  @Override
  public Scalar value(Tensor state, Tensor action) {
    Tensor stateAction = StateAction.key(state, action);
    return w.dot(featureMapper.getFeature(stateAction)).Get();
  }

  /** not needed! */
  @Override
  public void assign(Tensor state, Tensor action, Scalar value) {
    return;
  }

  @Override
  public QsaInterface copy() {
    return new FeatureQsaAdapter(w, featureMapper);
  }
}
