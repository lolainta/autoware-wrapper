class AvError(Exception):
    pass


class RouteNotFoundError(AvError):
    pass


class ResetError(AvError):
    pass


class LocalizationTimeoutError(ResetError):
    pass


class PlanningTimeoutError(ResetError):
    pass
